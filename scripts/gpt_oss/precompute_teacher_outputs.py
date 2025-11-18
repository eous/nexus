#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
GPT-OSS Teacher Output Precomputation

Precompute teacher model outputs for offline distillation.

This script generates and saves teacher model outputs to enable:
1. Offline distillation (teacher not needed during training, saves ~80GB VRAM)
2. Faster iteration (precompute once, train many student configs)
3. Reproducible experiments (same teacher outputs across runs)
4. DDP support (use 2+ GPUs to speed up precomputation)

Storage format: Parquet files with compression
- One file per step for easy random access
- Top-K logits instead of full vocabulary (200K → 100 = 2000x smaller)
- FP16 precision with zstd compression
- Typical size: 10K steps @ batch=4, seq=1024 → ~15-25 GB

Usage:
    # Basic: 10K steps from nemotron datasets
    python precompute_teacher_outputs.py \\
      --teacher-model /mnt/models/gpt-oss-120b \\
      --output-dir precomputed/teacher_10k \\
      --num-steps 10000 \\
      --batch-size 4 \\
      --seq-len 1024

    # Custom datasets and splits
    python precompute_teacher_outputs.py \\
      --teacher-model /mnt/models/gpt-oss-120b \\
      --output-dir precomputed/custom \\
      --num-steps 5000 \\
      --datasets nemotron-code nemotron-math c4 \\
      --dataset-splits train train validation \\
      --batch-size 8 \\
      --seq-len 512
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

# For Parquet writing
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required for Parquet support")
    print("Install with: pip install pyarrow")
    sys.exit(1)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Accelerate for optional DDP support
try:
    from accelerate import Accelerator
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False

# Local imports
from dataset import create_custom_dataset


def precompute_teacher_outputs(
    teacher_model,
    dataloader,
    output_dir: Path,
    num_steps: int,
    top_k: int = 100,
    compression: str = 'zstd',
    device: str = 'cuda:0',
    accelerator=None,
    validate_every: int = 1000,
    dataset_info: Optional[Dict] = None
):
    """
    Precompute teacher outputs and save to Parquet files.

    Supports DDP: Each rank saves different steps (interleaved).
    - Rank 0: steps 0, 2, 4, 6, ...
    - Rank 1: steps 1, 3, 5, 7, ...

    Args:
        teacher_model: Teacher model
        dataloader: DataLoader for input data (will be distributed if accelerator provided)
        output_dir: Directory to save precomputed outputs
        num_steps: Number of steps to precompute (total across all ranks)
        top_k: Number of top logits to store (default: 100)
        compression: Parquet compression ('zstd', 'snappy', 'gzip', or None)
        device: Device for teacher model
        accelerator: Optional Accelerator for DDP
        validate_every: Validate outputs every N steps

    Storage format:
        outputs/
            step_00000.parquet  # Step 0 data
            step_00001.parquet  # Step 1 data
            ...
            metadata.json       # Dataset info, model config, etc.
    """
    # DDP configuration
    is_main_process = (accelerator is None) or accelerator.is_main_process
    rank = accelerator.process_index if accelerator is not None else 0
    num_processes = accelerator.num_processes if accelerator is not None else 1

    output_dir = Path(output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Synchronize after directory creation
    if accelerator is not None:
        accelerator.wait_for_everyone()

    if is_main_process:
        print(f"\nPrecomputing {num_steps} teacher outputs...")
        print(f"  Output directory: {output_dir}")
        print(f"  Top-K logits: {top_k} (out of {teacher_model.config.vocab_size})")
        print(f"  Compression: {compression or 'none'}")
        if num_processes > 1:
            print(f"  DDP: {num_processes} processes (each rank saves different steps)")
            print(f"  Rank {rank} will save steps: 0, {num_processes}, {num_processes*2}, ...")
        else:
            print(f"  Device: {device}")

    teacher_model.eval()

    # Collect metadata
    metadata = {
        'teacher_model': str(teacher_model.config._name_or_path) if hasattr(teacher_model.config, '_name_or_path') else 'unknown',
        'num_steps': num_steps,
        'top_k': top_k,
        'vocab_size': teacher_model.config.vocab_size,
        'num_layers': teacher_model.config.num_hidden_layers,
        'compression': compression,
        'timestamp': datetime.now().isoformat(),
        'format_version': '1.1',  # Bumped to 1.1 (added attention_mask)
        'attention_mask_included': True  # New in format v1.1
    }

    # Add dataset configuration for validation during training
    if dataset_info:
        metadata.update(dataset_info)

    # Statistics tracking
    total_size_bytes = 0
    step_sizes = []

    if is_main_process:
        print(f"\nStarting precomputation...")
        print(f"  Target steps: {num_steps:,} (total across all ranks)")
        if num_processes > 1:
            print(f"  Steps per rank: ~{num_steps // num_processes:,}")

    # Progress bar only on main process, track total steps across all ranks
    pbar = tqdm(total=num_steps, desc="Precomputing", disable=not is_main_process)

    local_step = 0  # Steps processed by this rank
    for batch_idx, batch in enumerate(dataloader):
        # Calculate global step index for this batch
        # In DDP, each rank processes different batches:
        # - Dataloader is already distributed by Accelerate
        # - Each batch has a unique global index
        # - Global step = local_step * num_processes + rank
        global_step = local_step * num_processes + rank

        if global_step >= num_steps:
            break

        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = teacher_model(**batch, output_router_logits=True)

            # Extract top-K logits (vocabulary compression)
            # Shape: (batch, seq_len, vocab_size) → (batch, seq_len, top_k)
            top_k_values, top_k_indices = torch.topk(outputs.logits, k=top_k, dim=-1)

            # Convert to numpy (CPU) with FP16 to save space
            top_k_values_np = top_k_values.cpu().numpy().astype(np.float16)
            top_k_indices_np = top_k_indices.cpu().numpy().astype(np.int32)

            # Extract router logits
            # Shape: List[Tensor(batch, seq_len, num_experts)] → (num_layers, batch, seq_len, num_experts)
            router_logits_stacked = torch.stack(outputs.router_logits).cpu().numpy().astype(np.float16)

            # Get input_ids and attention_mask for reference
            input_ids_np = batch['input_ids'].cpu().numpy().astype(np.int32)
            attention_mask_np = batch['attention_mask'].cpu().numpy().astype(np.int8)

            # Compute hash for sample verification during training
            import hashlib
            hash_obj = hashlib.sha256()
            hash_obj.update(input_ids_np.tobytes())
            input_ids_hash = hash_obj.hexdigest()[:16]

        # Prepare data for Parquet
        # Parquet works best with flat tables, so we'll flatten arrays into binary blobs
        batch_size, seq_len, _ = top_k_values_np.shape
        num_layers = router_logits_stacked.shape[0]

        # Store as a single-row table with metadata + binary columns
        table_data = {
            'step': [global_step],  # Save with global step index
            'batch_size': [batch_size],
            'seq_len': [seq_len],
            'top_k': [top_k],
            'num_layers': [num_layers],
            'input_ids_hash': [input_ids_hash],  # For runtime verification
            # Binary blobs (Parquet handles these efficiently)
            'input_ids': [input_ids_np.tobytes()],
            'attention_mask': [attention_mask_np.tobytes()],
            'top_k_values': [top_k_values_np.tobytes()],
            'top_k_indices': [top_k_indices_np.tobytes()],
            'router_logits': [router_logits_stacked.tobytes()],
        }

        # Create PyArrow table
        pa_table = pa.table(table_data)

        # Write to Parquet file (use global_step for filename)
        step_file = output_dir / f"step_{global_step:05d}.parquet"
        pq.write_table(pa_table, step_file, compression=compression)

        # Track storage
        file_size = step_file.stat().st_size
        total_size_bytes += file_size
        step_sizes.append(file_size)

        # Increment local step counter (this rank's step count)
        local_step += 1

        # Update progress bar (only main process, but track global progress)
        if is_main_process:
            pbar.update(num_processes)  # Each batch represents num_processes global steps

        # Progress update every 100 local steps
        if local_step % 100 == 0 and is_main_process:
            avg_size_mb = np.mean(step_sizes) / 1e6
            estimated_total_gb = (avg_size_mb * num_steps) / 1e3
            print(f"  Rank {rank} - Step {global_step}/{num_steps}: Avg size {avg_size_mb:.2f} MB/step, "
                  f"estimated total {estimated_total_gb:.2f} GB")

    # END OF FOR LOOP - Dataset exhausted or reached num_steps

    # Close progress bar
    pbar.close()

    # Synchronize all ranks before checking completion
    if accelerator is not None:
        accelerator.wait_for_everyone()

    # Calculate actual steps completed
    total_steps_saved = local_step
    global_steps_completed = total_steps_saved * num_processes

    # Check if we got all requested steps
    if global_steps_completed < num_steps:
        # Dataset exhausted before reaching requested steps - ERROR OUT
        if is_main_process:
            print(f"\n{'='*80}")
            print(f"ERROR: Dataset exhausted before reaching requested steps")
            print(f"{'='*80}")
            print(f"  Requested: {num_steps:,} steps")
            print(f"  Completed: {global_steps_completed:,} steps")
            print(f"  Shortfall: {num_steps - global_steps_completed:,} steps")
            if num_processes > 1:
                print(f"  Per-rank: {total_steps_saved:,} steps (rank {rank})")
            print(f"\nSolutions:")
            print(f"  1. Reduce --num-steps to {global_steps_completed}")
            print(f"  2. Increase --max-samples (current: {metadata.get('max_samples', 'unknown')})")
            print(f"  3. Use different/larger datasets")
            print(f"\nPartial outputs saved to: {output_dir}")
            print(f"{'='*80}\n")

        # Save metadata with actual steps before erroring out
        if is_main_process:
            metadata['actual_steps'] = global_steps_completed
            metadata['requested_steps'] = num_steps
            metadata['incomplete'] = True
            metadata['num_processes'] = num_processes
            metadata['per_rank_steps'] = total_steps_saved
            with open(output_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved (marked as incomplete)")

        # ERROR OUT
        raise RuntimeError(
            f"Dataset exhausted after {global_steps_completed:,} steps (requested {num_steps:,}). "
            f"Increase --max-samples or reduce --num-steps."
        )

    # SUCCESS PATH: Save metadata (only on main process)
    if is_main_process:
        # Per-rank statistics
        metadata['per_rank_size_gb'] = total_size_bytes / 1e9
        metadata['avg_step_size_mb'] = np.mean(step_sizes) / 1e6 if step_sizes else 0

        # Estimate total across all ranks
        estimated_total_gb = (total_size_bytes / 1e9) * num_processes
        metadata['estimated_total_size_gb'] = estimated_total_gb

        metadata['first_batch_shape'] = {
            'batch_size': batch_size,
            'seq_len': seq_len
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'='*80}")
        print(f"PRECOMPUTATION COMPLETE")
        print(f"{'='*80}")
        if num_processes > 1:
            print(f"  DDP: {num_processes} processes")
            print(f"  Steps per rank: {total_steps_saved:,}")
            print(f"  Total steps: {num_steps:,}")
            print(f"  Estimated total size: {estimated_total_gb:.2f} GB")
        else:
            print(f"  Total steps: {total_steps_saved:,}")
            print(f"  Total size: {total_size_bytes / 1e9:.2f} GB")
        print(f"  Avg per step: {np.mean(step_sizes) / 1e6:.2f} MB")
        print(f"  Metadata: {output_dir / 'metadata.json'}")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Precompute teacher outputs for offline distillation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: 10K steps from default nemotron datasets
  python precompute_teacher_outputs.py \\
    --teacher-model /mnt/models/gpt-oss-120b \\
    --output-dir precomputed/teacher_10k \\
    --num-steps 10000

  # Custom configuration
  python precompute_teacher_outputs.py \\
    --teacher-model /mnt/models/gpt-oss-120b \\
    --output-dir precomputed/custom \\
    --num-steps 5000 \\
    --datasets nemotron-code nemotron-math \\
    --dataset-splits train train \\
    --batch-size 8 \\
    --seq-len 512

  # High compression for storage-constrained setups
  python precompute_teacher_outputs.py \\
    --teacher-model /mnt/models/gpt-oss-120b \\
    --output-dir precomputed/compressed \\
    --num-steps 10000 \\
    --top-k 50 \\
    --compression zstd
        """
    )

    # Model arguments
    parser.add_argument("--teacher-model", type=str, required=True,
                       help="Path to teacher model")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for teacher model (default: cuda:0)")

    # Output arguments
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save precomputed outputs")
    parser.add_argument("--num-steps", type=int, required=True,
                       help="Number of steps to precompute")

    # Data arguments
    parser.add_argument("--datasets", type=str, nargs='+',
                       default=['nemotron-code', 'nemotron-math', 'nemotron-tool'],
                       help="Datasets to use (default: nemotron-code nemotron-math nemotron-tool). "
                            "Options: nemotron-code, nemotron-math, nemotron-tool, c4")
    parser.add_argument("--dataset-splits", type=str, nargs='+',
                       default=['train', 'train', 'train'],
                       help="Split for each dataset (default: train train train)")
    parser.add_argument("--local-nemotron-code", type=str, default=None,
                       help="Path to local Nemotron code dataset")
    parser.add_argument("--local-nemotron-math", type=str, default=None,
                       help="Path to local Nemotron math dataset")
    parser.add_argument("--local-nemotron-tool", type=str, default=None,
                       help="Path to local Nemotron tool_calling dataset")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size (default: 4)")
    parser.add_argument("--seq-len", type=int, default=1024,
                       help="Sequence length (default: 1024)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum samples per dataset (default: None = unlimited streaming). "
                            "For precomputation, ensure this is >= num_steps * batch_size to avoid exhaustion.")

    # Compression arguments
    parser.add_argument("--top-k", type=int, default=100,
                       help="Number of top logits to store (default: 100, full vocab=201088)")
    parser.add_argument("--compression", type=str, default='zstd',
                       choices=['zstd', 'snappy', 'gzip', 'none'],
                       help="Parquet compression algorithm (default: zstd)")

    # Validation
    parser.add_argument("--validate-every", type=int, default=1000,
                       help="Validate outputs every N steps (default: 1000)")

    args = parser.parse_args()

    # Validate arguments
    if len(args.datasets) != len(args.dataset_splits):
        parser.error(f"Number of datasets ({len(args.datasets)}) must match number of splits ({len(args.dataset_splits)})")

    if args.compression == 'none':
        args.compression = None

    print("="*80)
    print("Teacher Output Precomputation for Offline Distillation")
    print("="*80)
    print(f"Teacher model: {args.teacher_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Steps to precompute: {args.num_steps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'unlimited (streaming)'}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Splits: {', '.join(args.dataset_splits)}")
    print("="*80)

    # Validate device
    if 'cuda' in args.device:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but --device specifies CUDA")
        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Device {args.device} not available (only {torch.cuda.device_count()} GPU(s))")

    # Check MXFP4 requires CUDA
    if args.device == 'cpu':
        print("WARNING: CPU device specified. MXFP4 models require CUDA.")
        print("This will likely fail. Use --device cuda:0")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # Load teacher model
    print("\nLoading teacher model...")
    teacher_device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        device_map={"": teacher_device_id},
        torch_dtype=torch.bfloat16
    )
    teacher_model.eval()
    print(f"  ✓ Model loaded on {args.device}")
    print(f"  Config: {teacher_model.config.num_hidden_layers} layers, "
          f"{teacher_model.config.vocab_size} vocab size")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    print(f"  ✓ Tokenizer loaded (vocab_size: {tokenizer.vocab_size})")

    # Create dataloader using dataset module
    print("\nCreating dataloader...")

    from torch.utils.data import DataLoader
    import logging

    # Setup logger for dataset module
    logger = logging.getLogger('dataset')
    logger.setLevel(logging.INFO)

    # Build local paths dict
    local_paths = {
        'code': args.local_nemotron_code,
        'math': args.local_nemotron_math,
        'tool': args.local_nemotron_tool
    }

    # Create custom interleaved dataset
    stream_dataset = create_custom_dataset(
        tokenizer=tokenizer,
        datasets=args.datasets,
        splits=args.dataset_splits,
        local_paths=local_paths,
        max_length=args.seq_len,
        max_samples=args.max_samples,
        logger=logger
    )

    # Create DataLoader
    dataloader = DataLoader(
        stream_dataset,
        batch_size=args.batch_size,
        num_workers=0,  # Keep 0 for compatibility with streaming
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"  ✓ DataLoader created (streaming mode)")

    # Initialize Accelerator for DDP if running with accelerate launch
    accelerator = None
    if ACCELERATE_AVAILABLE:
        try:
            # Check if we're in a distributed environment
            import os
            if 'ACCELERATE_TORCH_DEVICE' in os.environ or 'LOCAL_RANK' in os.environ:
                accelerator = Accelerator()
                is_main = accelerator.is_main_process
                if is_main:
                    print(f"\n{'='*80}")
                    print(f"DDP ENABLED via Accelerate")
                    print(f"{'='*80}")
                    print(f"  Processes: {accelerator.num_processes}")
                    print(f"  Rank: {accelerator.process_index}")
                    print(f"  Device: {accelerator.device}")
                    print(f"  Each rank will save different steps (interleaved)")
                    print(f"{'='*80}\n")

                # Prepare dataloader for DDP
                dataloader = accelerator.prepare(dataloader)
        except Exception as e:
            print(f"Note: Accelerate available but not in distributed mode: {e}")

    # Prepare dataset info for metadata
    dataset_info = {
        'datasets': args.datasets,
        'dataset_splits': args.dataset_splits,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'max_samples': args.max_samples
    }

    # Call the precompute_teacher_outputs function
    precompute_teacher_outputs(
        teacher_model=teacher_model,
        dataloader=dataloader,
        output_dir=Path(args.output_dir),
        num_steps=args.num_steps,
        top_k=args.top_k,
        compression=args.compression,
        device=args.device,
        accelerator=accelerator,
        validate_every=args.validate_every,
        dataset_info=dataset_info
    )


# OLD INLINE CODE REMOVED - Now using the function above
# The code below (lines 492-605 in original) was duplicate implementation
# It has been replaced with the function call above

"""
REMOVED DUPLICATE CODE (was lines 475-605):
This section contained inline precomputation logic that duplicated the
precompute_teacher_outputs() function. Refactored to use the function instead.
"""


if __name__ == '__main__':
    main()
