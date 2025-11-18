#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
Precomputed Teacher Outputs Loader

Loader for precomputed teacher outputs.

Efficiently loads teacher model outputs from Parquet files for offline distillation.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required for loading precomputed outputs")
    print("Install with: pip install pyarrow")
    raise


class PrecomputedTeacherOutputs:
    """
    Container for precomputed teacher outputs.

    Mimics the interface of model outputs for drop-in replacement.
    """

    def __init__(self, logits: torch.Tensor, router_logits: list, input_ids_hash: Optional[str] = None):
        """
        Initialize precomputed outputs.

        Args:
            logits: Reconstructed logits tensor (batch, seq_len, vocab_size)
                   Note: For top-K storage, this is sparse (only top-K values, rest are -inf)
            router_logits: List of router logit tensors, one per layer
            input_ids_hash: Hash of input_ids for verification (None if not available)
        """
        self.logits = logits
        self.router_logits = router_logits
        self.input_ids_hash = input_ids_hash


class PrecomputedTeacherLoader:
    """
    Loader for precomputed teacher outputs from Parquet files.

    Loads teacher model outputs on-demand for offline distillation.
    """

    def __init__(self, precomputed_dir: str, device: str = 'cuda:0'):
        """
        Initialize loader.

        Args:
            precomputed_dir: Directory containing precomputed outputs
            device: Device to load tensors to
        """
        self.precomputed_dir = Path(precomputed_dir)
        self.device = device

        # Load metadata
        metadata_path = self.precomputed_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Validate format version
        self.format_version = self.metadata.get('format_version')
        if self.format_version != '1.1':
            raise ValueError(
                f"Incompatible precomputed data format: {self.format_version}\n"
                f"This version of NEXUS requires format v1.1.\n"
                f"Please regenerate precomputed data with the current version."
            )

        # Check for incomplete data
        if self.metadata.get('incomplete'):
            raise ValueError(
                f"Precomputed data is incomplete!\n"
                f"  Requested: {self.metadata.get('requested_steps', 'unknown')} steps\n"
                f"  Actual: {self.metadata.get('actual_steps', 'unknown')} steps\n"
                f"Please regenerate with sufficient data."
            )

        # Extract required fields
        self.num_steps = self.metadata['num_steps']
        self.top_k = self.metadata['top_k']
        self.vocab_size = self.metadata['vocab_size']
        self.num_layers = self.metadata['num_layers']

        print(f"✓ PrecomputedTeacherLoader initialized")
        print(f"  Directory: {self.precomputed_dir}")
        print(f"  Format version: {self.format_version}")
        print(f"  Steps available: {self.num_steps:,}")
        print(f"  Top-K: {self.top_k} (full vocab: {self.vocab_size})")
        print(f"  Layers: {self.num_layers}")
        print(f"  Total size: {self.metadata['total_size_gb']:.2f} GB")
        print(f"  Avg per step: {self.metadata['avg_step_size_mb']:.2f} MB")
        print(f"  Datasets: {', '.join(self.metadata['datasets'])}")
        print(f"  Batch size: {self.metadata['batch_size']}")
        print(f"  Seq length: {self.metadata['seq_len']}")

    def load_step(self, step: int) -> Tuple[Dict[str, torch.Tensor], PrecomputedTeacherOutputs]:
        """
        Load precomputed teacher outputs for a specific step.

        Args:
            step: Step number to load

        Returns:
            (batch, teacher_outputs) where:
                batch: Dict with 'input_ids', 'attention_mask', 'labels' (same format as dataloader)
                teacher_outputs: PrecomputedTeacherOutputs with .logits and .router_logits
        """
        if step >= self.num_steps:
            raise ValueError(f"Step {step} not available (only {self.num_steps} steps precomputed)")

        # Load Parquet file
        step_file = self.precomputed_dir / f"step_{step:05d}.parquet"
        if not step_file.exists():
            raise FileNotFoundError(f"Step file not found: {step_file}")

        table = pq.read_table(step_file)
        data = table.to_pydict()

        # Extract metadata
        batch_size = data['batch_size'][0]
        seq_len = data['seq_len'][0]
        top_k = data['top_k'][0]
        num_layers = data['num_layers'][0]

        # Extract hash for verification (required in format v1.1)
        input_ids_hash = data['input_ids_hash'][0]
        if isinstance(input_ids_hash, bytes):
            input_ids_hash = input_ids_hash.decode('utf-8')

        # Reconstruct arrays from binary blobs
        input_ids = np.frombuffer(data['input_ids'][0], dtype=np.int32).reshape(batch_size, seq_len)
        attention_mask = np.frombuffer(data['attention_mask'][0], dtype=np.int8).reshape(batch_size, seq_len)
        top_k_values = np.frombuffer(data['top_k_values'][0], dtype=np.float16).reshape(batch_size, seq_len, top_k)
        top_k_indices = np.frombuffer(data['top_k_indices'][0], dtype=np.int32).reshape(batch_size, seq_len, top_k)
        router_logits_flat = np.frombuffer(data['router_logits'][0], dtype=np.float16)
        router_logits_array = router_logits_flat.reshape(num_layers, batch_size, seq_len, 128)

        # Convert to tensors (copy arrays since PyArrow returns read-only)
        # CRITICAL: input_ids must be Long (INT64) for cross_entropy, not Int (INT32)
        input_ids_tensor = torch.from_numpy(input_ids.copy()).to(torch.long).to(self.device)
        attention_mask_tensor = torch.from_numpy(attention_mask.copy()).to(self.device)

        # Reconstruct sparse logits tensor
        # Create a full-vocab tensor with -inf everywhere except top-K positions
        logits_reconstructed = torch.full(
            (batch_size, seq_len, self.vocab_size),
            float('-inf'),
            dtype=torch.float32,
            device=self.device
        )

        # Fill in top-K values (copy arrays since PyArrow returns read-only)
        top_k_values_tensor = torch.from_numpy(top_k_values.copy()).to(torch.float32).to(self.device)
        top_k_indices_tensor = torch.from_numpy(top_k_indices.copy()).to(torch.long).to(self.device)

        # Scatter top-K values into full logits tensor
        for b in range(batch_size):
            for s in range(seq_len):
                logits_reconstructed[b, s, top_k_indices_tensor[b, s]] = top_k_values_tensor[b, s]

        # Convert router logits to list of tensors (one per layer, copy for writability)
        router_logits_list = []
        for layer_idx in range(num_layers):
            router_tensor = torch.from_numpy(router_logits_array[layer_idx].copy()).to(torch.float32).to(self.device)
            router_logits_list.append(router_tensor)

        # Create labels (shift input_ids by 1 for causal LM)
        # Standard approach: labels[i] = input_ids[i+1]
        # We need to handle this carefully to maintain consistency with training
        labels = input_ids_tensor.clone()
        # Note: The model's forward pass will handle the shifting internally
        # We just need to provide labels that match the input_ids

        # Create batch dict (same format as dataloader)
        batch = {
            'input_ids': input_ids_tensor,
            'attention_mask': attention_mask_tensor,
            'labels': labels  # Model will use this for LM loss
        }

        # Create output container
        teacher_outputs = PrecomputedTeacherOutputs(
            logits=logits_reconstructed,
            router_logits=router_logits_list,
            input_ids_hash=input_ids_hash
        )

        return batch, teacher_outputs

    def validate(self) -> bool:
        """
        Validate precomputed outputs by loading first step.

        Returns:
            True if validation passed
        """
        print("\nValidating precomputed outputs...")

        try:
            batch, outputs = self.load_step(0)
            print(f"  ✓ Step 0 loaded successfully")
            print(f"    Batch keys: {list(batch.keys())}")
            print(f"    Input IDs shape: {batch['input_ids'].shape}")
            print(f"    Attention mask shape: {batch['attention_mask'].shape}")
            print(f"    Labels shape: {batch['labels'].shape}")
            print(f"    Teacher logits shape: {outputs.logits.shape}")
            print(f"    Teacher router logits: {len(outputs.router_logits)} layers")

            # Check for NaN/Inf
            if torch.isnan(outputs.logits).any():
                print("  ✗ WARNING: Logits contain NaN")
                return False
            if torch.isinf(outputs.logits).sum() < outputs.logits.numel() * 0.9:
                # Should be mostly -inf (sparse storage), but some real values
                print(f"  ✗ WARNING: Unexpected logits distribution")
                return False

            print(f"  ✓ Validation passed")
            return True

        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            return False

    def get_metadata(self) -> Dict:
        """Get metadata dictionary."""
        return self.metadata.copy()

    def iter_steps(self, rank: int = 0, num_processes: int = 1):
        """
        Iterate through steps for a specific rank in DDP mode.

        This is the recommended way to iterate through precomputed data during training.
        Each rank processes different steps in an interleaved pattern:
        - Rank 0: steps 0, 2, 4, 6, ...
        - Rank 1: steps 1, 3, 5, 7, ...

        Args:
            rank: Process rank (0 to num_processes-1)
            num_processes: Total number of processes (1 for non-DDP)

        Yields:
            (precomputed_step, batch, teacher_outputs) tuples where:
                precomputed_step: Global step number in precomputed data
                batch: Dict with 'input_ids', 'attention_mask', 'labels'
                teacher_outputs: PrecomputedTeacherOutputs with logits and router_logits

        Example:
            >>> loader = PrecomputedTeacherLoader('teacher_outputs/')
            >>> for step, batch, outputs in loader.iter_steps(rank=0, num_processes=2):
            >>>     student_outputs = student_model(**batch)
            >>>     loss = compute_loss(student_outputs, outputs)
        """
        local_step = 0
        while True:
            # Calculate which precomputed step this rank should load
            precomputed_step = local_step * num_processes + rank

            # Check if we've exhausted the precomputed data
            if precomputed_step >= self.num_steps:
                break

            # Load batch and teacher outputs
            batch, teacher_outputs = self.load_step(precomputed_step)

            yield precomputed_step, batch, teacher_outputs

            local_step += 1


def reconstruct_full_logits_from_topk(top_k_values: torch.Tensor,
                                      top_k_indices: torch.Tensor,
                                      vocab_size: int,
                                      device: str = 'cuda') -> torch.Tensor:
    """
    Reconstruct full vocabulary logits from top-K values.

    Args:
        top_k_values: Top-K logit values (batch, seq_len, top_k)
        top_k_indices: Top-K indices (batch, seq_len, top_k)
        vocab_size: Full vocabulary size
        device: Device for output tensor

    Returns:
        Sparse logits tensor (batch, seq_len, vocab_size) with -inf for non-top-K
    """
    batch_size, seq_len, top_k = top_k_values.shape

    # Create tensor filled with -inf
    logits = torch.full(
        (batch_size, seq_len, vocab_size),
        float('-inf'),
        dtype=top_k_values.dtype,
        device=device
    )

    # Scatter top-K values (vectorized version)
    # This is faster than the loop version in load_step
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1).expand(-1, seq_len, top_k)
    seq_indices = torch.arange(seq_len, device=device).view(1, -1, 1).expand(batch_size, -1, top_k)

    logits[batch_indices, seq_indices, top_k_indices] = top_k_values

    return logits
