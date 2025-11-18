#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
GPT-OSS Offline Distillation Training with DDP

Trains student model using precomputed teacher outputs with Distributed Data Parallel.
Teacher model is NOT loaded (saves ~80GB VRAM). Precomputed outputs are loaded from disk.

Key features:
- Offline distillation: Uses precomputed teacher outputs (no teacher in memory)
- DDP: Multi-GPU training with Accelerate
- SFA: Rolling anchor merging (DDP-safe with broadcasting)
- Memory efficient: Only student model in memory

================================================================================
HARDWARE REQUIREMENTS
================================================================================

- 2+ GPUs for DDP (tested with RTX Pro 6000 Blackwell)
- ~50-60GB VRAM per GPU for student model
- Fast storage for precomputed outputs

Memory per GPU:
- Student model: ~70GB (MXFP4 quantized routed experts + BF16 shared expert)
- Training overhead: ~5-10GB (gradients + optimizer states)
- Peak: ~75-80GB per GPU (fits on 98GB cards with headroom)
- NO teacher model (~80GB saved vs online distillation)

================================================================================
USAGE
================================================================================

Step 1: Precompute teacher outputs (separate script, run once):
    python precompute_teacher.py \\
        --teacher-model /path/to/gpt-oss-120b \\
        --output teacher_outputs/ \\
        --num-steps 100000

Step 2: Train with DDP using precomputed outputs:
    accelerate launch --num_processes=2 train_offline.py \\
        --precomputed-teacher teacher_outputs/ \\
        --student-model /path/to/gpt-oss-120b-nexus \\
        --output-dir outputs/ \\
        --max-steps 10000

================================================================================
"""

# Standard library imports
import argparse
import contextlib
import copy
import json
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Third-party core imports
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

# HuggingFace imports
from transformers import AutoTokenizer, set_seed

# Accelerate (required for DDP)
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

# Local imports - shared modules
# NOTE: No dataset import needed! We iterate through precomputed files directly.
from distillation import compute_loss_with_kl
from sfa import (
    load_trainable_weights_from_checkpoint,
    merge_with_anchor,
    save_trainable_weights_safetensors
)
from model_utils import (
    load_student_model,
    prepare_model_for_stage0,
    set_deterministic_training
)
from scheduler import GPTOSSTrainingScheduler, create_optimizer_with_bias_groups

def train(args):
    """
    Offline distillation training loop with DDP and SFA.

    This function trains the student model using precomputed teacher outputs with
    Distributed Data Parallel across multiple GPUs. Teacher model is NOT loaded.

    Key Features:
    - Offline distillation: Uses precomputed teacher outputs from disk
    - DDP: Multi-GPU training with Accelerate
    - Rolling anchor SFA: Periodically merge with previous checkpoint (DDP-safe)
    - Memory-efficient: Only student model in memory (~909M trainable params)

    Training Flow:
    1. Setup Accelerate for DDP
    2. Load precomputed teacher outputs
    3. Load student model (distributed across GPUs)
    4. Training loop with gradient accumulation and DDP synchronization
    5. DDP-safe SFA merging with broadcasting
    6. Checkpointing (main process only)

    Args:
        args: Parsed command-line arguments
              Required: precomputed_teacher, student_model, max_steps, output_dir

    Memory Usage (per GPU for GPT-OSS-120B):
        - Student model: ~70GB (MXFP4 quantized)
        - Training overhead: ~5-10GB (gradients + optimizer states)
        - Peak: ~75-80GB per GPU
        - NO teacher model (~80GB saved vs online)

    Returns:
        None (saves checkpoints and metrics to disk)
    """

    print("="*80)
    print("OFFLINE Distillation Training with DDP + SFA")
    print("="*80)

    # Setup Accelerate for DDP (always required for this script)
    # CRITICAL: Set environment variable for DDP to handle frozen router biases
    # This must be set BEFORE Accelerator is created
    # Skip if router is frozen (no dynamic changes)
    if args.use_advanced_scheduler and not args.freeze_router:
        os.environ['ACCELERATE_USE_FIND_UNUSED_PARAMETERS'] = 'true'
        print("  ✓ Configured DDP for dynamic parameter freezing (find_unused_parameters=True)")

    # Initialize Accelerator with gradient accumulation support
    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None  # We handle our own logging
    )

    print(f"\n{'='*80}")
    print(f"DISTRIBUTED DATA PARALLEL (DDP) via Accelerate")
    print(f"{'='*80}")
    print(f"  Process: {accelerator.process_index}/{accelerator.num_processes}")
    print(f"  Device: {accelerator.device}")
    print(f"  Main process: {accelerator.is_main_process}")
    print(f"  Mixed precision: bf16")
    print(f"{'='*80}\n")

    # Determine if we should print (only main process in DDP)
    should_print = accelerator.is_main_process

    # Set random seed for reproducibility
    if args.seed is not None:
        if args.deterministic:
            set_deterministic_training(args.seed)
        else:
            # Set seed without full deterministic mode (faster)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            set_seed(args.seed)
            if should_print:
                print(f"Set random seed to {args.seed}")

    print(f"Temperature: {args.temperature} (initial)")
    print(f"KL Weight: {args.kl_weight}")

    # Conditional LR schedule info
    if args.use_advanced_scheduler:
        print(f"Learning Rate: {args.learning_rate}")
        print(f"  Scheduler: ADVANCED (DeepSeek-V3 style)")
        print(f"  LR Schedule: Warmup → Stable → Cosine Decay")
        if args.freeze_router:
            print(f"  Router: FROZEN (training shared expert only)")
        else:
            bias_freeze_pct = int((args.bias_freeze_ratio if args.bias_freeze_ratio is not None else args.warmup_ratio) * 100)
            print(f"  Router Bias: Slow updates → Freeze at {bias_freeze_pct}%")
        print(f"  Temperature: {args.temperature} → {args.final_temperature}")
        if args.kl_weight_warmup:
            print(f"  KL Weight: Warmup 0.0 → 1.0")
    elif args.disable_sfa:
        print(f"Learning Rate: {args.learning_rate} (with LINEAR DECAY)")
        print(f"  LR Schedule: Warmup (10%) then linear decay to 0")
        print(f"  Mode: Standard training (SFA disabled)")
    else:
        print(f"Learning Rate: {args.learning_rate} (CONSTANT after warmup)")
        print(f"  LR Schedule: Warmup (10%) then constant (no decay)")
        print(f"  Rationale: SFA merging provides regularization")

    print(f"Gradient Clipping: {args.gradient_clip_val}")
    print(f"Max Steps: [Calculated from precomputed data]")
    print(f"Checkpoint Interval: {args.checkpoint_interval}")
    print(f"Expected Layers: {args.num_layers}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Gradient Checkpointing: {'Enabled' if args.gradient_checkpointing else 'Disabled'}")
    print(f"Log Every N Steps: {args.log_every_n_steps}")

    # SFA settings (only show if enabled)
    if not args.disable_sfa:
        print(f"\nSFA Settings (ROLLING ANCHOR approach):")

        # Validate and sync merge interval with checkpoint interval
        if args.sfa_merge_interval > 0:
            if args.sfa_merge_interval != args.checkpoint_interval:
                print(f"  ⚠️  WARNING: sfa_merge_interval ({args.sfa_merge_interval}) != checkpoint_interval ({args.checkpoint_interval})")
                print(f"  ⚠️  Setting sfa_merge_interval = {args.checkpoint_interval} (required for rolling anchor)")
                args.sfa_merge_interval = args.checkpoint_interval

        print(f"  Merge Interval: {args.sfa_merge_interval} steps (synced with checkpoint interval)")
        print(f"  Alpha (anchor weight): {args.sfa_alpha}")
        print(f"  Strategy: ROLLING ANCHOR")
        print(f"    - Step {args.checkpoint_interval}: Save checkpoint, NO merge (free training)")
        print(f"    - Step {args.checkpoint_interval*2}: Merge with checkpoint-{args.checkpoint_interval}, then save")
        print(f"    - Step {args.checkpoint_interval*3}: Merge with checkpoint-{args.checkpoint_interval*2}, then save")
        print(f"    - This allows gradual drift while preventing catastrophic forgetting")
        print(f"  Formula: merged = {args.sfa_alpha} * previous_checkpoint + {1-args.sfa_alpha} * current")
    else:
        print(f"\nSFA: DISABLED (standard training mode)")
        print(f"  No periodic merging")
        print(f"  No trainable_weights.safetensors files will be saved")
        print(f"  Using standard LR decay instead")

    print("="*80)

    # Setup output directory (only on main process in DDP)
    output_dir = Path(args.output_dir)
    if should_print:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = vars(args)
        config['timestamp'] = datetime.now().isoformat()
        config['method'] = 'SFA (Sequential Fine-tuning with Averaging)' if not args.disable_sfa else 'Standard Training (LR Decay)'
        config['sfa_enabled'] = not args.disable_sfa and args.sfa_merge_interval > 0
        config['sfa_disabled'] = args.disable_sfa
        config['use_ddp'] = True  # Always True for train_offline.py
        config['num_processes'] = accelerator.num_processes
        config['process_index'] = accelerator.process_index
        config['offline_distillation'] = True

        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Synchronize after directory creation
    accelerator.wait_for_everyone()

    # Validate CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for MXFP4 models but CUDA is not available.")

    num_gpus = torch.cuda.device_count()
    if should_print:
        print(f"✓ CUDA available: {num_gpus} GPU(s) detected")

    # Load precomputed teacher outputs (required for this script)
    from precomputed_loader import PrecomputedTeacherLoader

    if should_print:
        print("\n" + "="*80)
        print("LOADING PRECOMPUTED TEACHER OUTPUTS")
        print("="*80)
        print(f"Source: {args.precomputed_teacher}")

    precomputed_loader = PrecomputedTeacherLoader(
        precomputed_dir=args.precomputed_teacher,
        device="cuda"  # Accelerate will handle device placement
    )

    # Validate configuration matches (STRICT - fail on mismatch)
    precomp_meta = precomputed_loader.get_metadata()
    precomp_batch_size = precomp_meta.get('first_batch_shape', {}).get('batch_size')
    precomp_seq_len = precomp_meta.get('first_batch_shape', {}).get('seq_len')

    if precomp_batch_size != args.batch_size:
        if should_print:
            print(f"\n✗ ERROR: Batch size mismatch!")
            print(f"  Precomputed: batch_size={precomp_batch_size}")
            print(f"  Training:    batch_size={args.batch_size}")
            print(f"\nSolution: Either")
            print(f"  1. Regenerate precomputed outputs with --batch-size {args.batch_size}")
            print(f"  2. Change training --batch-size to {precomp_batch_size}")
        raise ValueError(f"Batch size mismatch: precomputed={precomp_batch_size}, training={args.batch_size}")

    if precomp_seq_len != args.seq_len:
        if should_print:
            print(f"\n✗ ERROR: Sequence length mismatch!")
            print(f"  Precomputed: seq_len={precomp_seq_len}")
            print(f"  Training:    seq_len={args.seq_len}")
            print(f"\nSolution: Either")
            print(f"  1. Regenerate precomputed outputs with --seq-len {args.seq_len}")
            print(f"  2. Change training --seq-len to {precomp_seq_len}")
        raise ValueError(f"Sequence length mismatch: precomputed={precomp_seq_len}, training={args.seq_len}")

    # Validate precomputed data integrity
    if not precomputed_loader.validate():
        raise RuntimeError("Precomputed data validation failed")

    # Calculate training steps from precomputed data
    # Each optimizer step requires gradient_accumulation_steps batches per process
    num_processes = accelerator.num_processes
    max_steps = precomputed_loader.num_steps // (args.gradient_accumulation_steps * num_processes)

    if max_steps == 0:
        if should_print:
            print(f"\n✗ ERROR: Insufficient precomputed steps!")
            print(f"  Precomputed: {precomputed_loader.num_steps:,} steps")
            print(f"  Need at least: {args.gradient_accumulation_steps * num_processes} steps")
            print(f"    Calculation: grad_accum ({args.gradient_accumulation_steps}) × processes ({num_processes})")
            print(f"\nSolution: Precompute more steps")
        raise RuntimeError("Insufficient precomputed steps")

    if should_print:
        print(f"✓ Precomputed teacher outputs ready ({precomputed_loader.num_steps:,} steps)")
        print(f"  Will train for: {max_steps:,} optimizer steps")
        print(f"    Calculation: {precomputed_loader.num_steps:,} ÷ ({args.gradient_accumulation_steps} × {num_processes}) = {max_steps:,}")
        unused_steps = precomputed_loader.num_steps - (max_steps * args.gradient_accumulation_steps * num_processes)
        if unused_steps > 0:
            print(f"  Note: {unused_steps:,} precomputed steps will be unused (not evenly divisible)")
        print(f"  Memory saved: ~80GB (teacher model not loaded)")
        print("="*80 + "\n")

    # Load student model for DDP training
    if should_print:
        print("Loading student model...")

    # Auto-detect model configuration
    from transformers import AutoConfig
    student_config = AutoConfig.from_pretrained(args.student_model, trust_remote_code=True)
    num_layers = student_config.num_hidden_layers

    if should_print:
        print(f"  Model variant detected:")
        print(f"    Layers: {num_layers}")
        print(f"    Hidden size: {student_config.hidden_size}")
        print(f"    Vocab size: {student_config.vocab_size}")
        if num_layers == 24:
            print(f"    → GPT-OSS 20B variant")
        elif num_layers == 36:
            print(f"    → GPT-OSS 120B variant")
        else:
            print(f"    → Custom GPT-OSS variant")

    # In DDP mode, each process loads to its own GPU

    # CRITICAL: In DDP mode, each process must load to its own GPU
    # Otherwise both processes try to load to GPU 0 simultaneously → OOM!
    local_rank = accelerator.local_process_index
    student_model = load_student_model(
        args.student_model,
        device_id=local_rank,
        use_ddp=True,
        local_rank=local_rank
    )

    # Prepare for Stage-0
    student_model = prepare_model_for_stage0(student_model, freeze_router=args.freeze_router)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        student_model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")

    # Track trainable parameters for SFA merging
    # Note: With rolling anchor approach, we don't store initial anchor - we load previous checkpoints
    trainable_param_names = {name for name, param in student_model.named_parameters() if param.requires_grad}
    print(f"\n✓ Tracked {len(trainable_param_names)} trainable parameter names for SFA merging")

    # Load tokenizer (only needed for SFA merging, not for data loading)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # NO DATALOADER NEEDED!
    # All data comes from precomputed teacher outputs.
    # We iterate directly through precomputed files using precomputed_loader.iter_steps()
    # This eliminates:
    # - Dataset configuration complexity
    # - Batch replacement logic
    # - Risk of dataset config mismatch
    # - Need for hash verification (though we keep it optional for debugging)

    # Import comprehensive scheduler (DeepSeek-V3 style)
    from scheduler import (
        create_optimizer_with_bias_groups,
        GPTOSSTrainingScheduler,
        print_schedule_summary
    )

    # Get list of trainable parameters (needed for gradient clipping in both modes)
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]

    # Setup optimizer with separate parameter groups for router biases
    # This enables differential learning rates (router biases get LR×0.001)
    if args.use_advanced_scheduler:
        if should_print:
            print(f"\n{'='*80}")
            print(f"ADVANCED SCHEDULER ENABLED (DeepSeek-V3 Style)")
            print(f"{'='*80}")

        # Skip bias groups if router is frozen (no router params to train)
        if args.freeze_router:
            if should_print:
                print(f"\n  Note: Router frozen - using simple optimizer (no bias groups)")
            optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        else:
            optimizer = create_optimizer_with_bias_groups(
                model=student_model,
                learning_rate=args.learning_rate,
                bias_lr_multiplier=args.bias_lr_multiplier,
                verbose=should_print  # Only print on main process in DDP
            )
    else:
        # Legacy: single parameter group
        if should_print:
            print(f"\n✓ Using standard optimizer (single parameter group)")
            print(f"✓ Collected {len(trainable_params)} trainable parameter tensors for optimization")
        optimizer = AdamW(trainable_params, lr=args.learning_rate)

    # Setup scheduler
    num_training_steps = max_steps  # Calculated from precomputed data

    if args.use_advanced_scheduler:
        # Router frozen: no bias freezing needed
        if args.freeze_router:
            bias_freeze_ratio = None  # No router params to freeze
            if should_print:
                print(f"\n  ✓ Router frozen - skipping bias freeze scheduling")
        else:
            # Default bias_freeze_ratio to warmup_ratio if not specified
            # For pretrained models, freezing biases right after warmup prevents destabilization
            bias_freeze_ratio = args.bias_freeze_ratio if args.bias_freeze_ratio is not None else args.warmup_ratio

            if should_print and args.bias_freeze_ratio is None:
                print(f"\n  ✓ bias_freeze_ratio not specified, defaulting to warmup_ratio ({args.warmup_ratio:.1%})")
                print(f"    Rationale: Freeze biases after warmup to prevent routing destabilization")

        # DeepSeek-V3 style: comprehensive scheduler with all advanced features
        scheduler = GPTOSSTrainingScheduler(
            model=student_model,
            optimizer=optimizer,
            total_steps=num_training_steps,
            peak_lr=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            stable_ratio=args.stable_ratio,
            final_lr_ratio=args.final_lr_ratio,
            bias_lr_multiplier=args.bias_lr_multiplier,
            bias_freeze_ratio=bias_freeze_ratio,  # None if router frozen
            initial_temperature=args.temperature,
            final_temperature=args.final_temperature,
            temperature_anneal_start_ratio=args.temperature_anneal_start,
            kl_weight_warmup=args.kl_weight_warmup
        )

        # Print schedule summary (only on main process in DDP)
        if should_print:
            print_schedule_summary(scheduler)

    elif args.disable_sfa:
        # Standard training: use linear LR decay
        from transformers import get_linear_schedule_with_warmup
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"\n✓ Using LINEAR LR DECAY schedule (standard training)")
        print(f"  Warmup steps: {num_warmup_steps} (10% of {num_training_steps})")
        print(f"  After warmup: LR decays linearly from {args.learning_rate} to 0")
        print(f"  Rationale: SFA disabled, using standard LR decay")

    else:
        # SFA mode: use constant LR after warmup
        from torch.optim.lr_scheduler import LambdaLR
        num_warmup_steps = int(0.1 * num_training_steps)

        def constant_schedule_with_warmup(current_step):
            """Warmup then constant LR (no decay)."""
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=constant_schedule_with_warmup)
        print(f"\n✓ Using CONSTANT LR schedule (SFA mode)")
        print(f"  Warmup steps: {num_warmup_steps} (10% of {num_training_steps})")
        print(f"  After warmup: LR stays at {args.learning_rate}")
        print(f"  Rationale: SFA merging provides regularization (decay not needed)")

    # Prepare components for DDP if using Accelerate
    if accelerator is not None:
        if should_print:
            print(f"\nPreparing components for distributed training...")

        # Accelerate will handle:
        # - Wrapping student_model in DDP
        # - Moving optimizer state to correct devices
        # Note: No dataloader to prepare (we iterate through precomputed files directly)
        student_model, optimizer = accelerator.prepare(
            student_model, optimizer
        )

        if should_print:
            print(f"  ✓ Model wrapped in DDP")
            print(f"  ✓ Optimizer prepared")
            print(f"  ✓ Precomputed data iteration (no dataloader needed)")
            print(f"  Effective batch size: {args.batch_size} × {accelerator.num_processes} GPUs "
                  f"× {args.gradient_accumulation_steps} accum = "
                  f"{args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps} samples/update")

    # Resume from checkpoint if specified
    global_step = 0
    all_metrics = []

    if args.resume_from_checkpoint:
        if should_print:
            print(f"\n{'='*80}")
            print(f"RESUMING FROM CHECKPOINT")
            print(f"{'='*80}")
            print(f"Checkpoint: {args.resume_from_checkpoint}")

        checkpoint_path = Path(args.resume_from_checkpoint)
        training_state_path = checkpoint_path / "training_state.pt"

        if not training_state_path.exists():
            raise FileNotFoundError(
                f"Training state not found: {training_state_path}\n"
                "This checkpoint may be from an older version without training state.\n"
                "You can only resume from checkpoints that include training_state.pt"
            )

        # Load training state
        training_state = torch.load(training_state_path, map_location='cpu')

        # Restore global step
        global_step = training_state['global_step']
        if should_print:
            print(f"  Resuming from step: {global_step}")
            print(f"  Remaining steps: {max_steps - global_step}")

        # Restore optimizer state
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        if should_print:
            print(f"  ✓ Optimizer state restored")

        # Restore scheduler state (if saved)
        if training_state['scheduler_state_dict'] is not None:
            if hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(training_state['scheduler_state_dict'])
                if should_print:
                    print(f"  ✓ Scheduler state restored")

        # Restore random states
        if 'random_state' in training_state:
            random_state = training_state['random_state']
            torch.set_rng_state(random_state['torch'])
            if random_state['torch_cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(random_state['torch_cuda'])
            np.random.set_state(random_state['numpy'])
            random.setstate(random_state['python'])
            if should_print:
                print(f"  ✓ Random states restored")

        # Load metrics history
        metrics_path = checkpoint_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                all_metrics = json.load(f)
            if should_print:
                print(f"  ✓ Loaded {len(all_metrics)} historical metrics")

        if should_print:
            print(f"{'='*80}\n")

    # Training loop initialization
    accumulation_steps = args.gradient_accumulation_steps

    # Restore partial accumulation state if resuming mid-accumulation
    # PERFORMANCE FIX: Use tensors for accumulation (avoid CUDA sync), but load as float from checkpoint
    if args.resume_from_checkpoint and 'accumulated_loss' in training_state:
        # Load as floats from checkpoint, will be converted to tensors on first accumulation
        accumulated_loss = training_state['accumulated_loss']
        accumulated_lm_loss = training_state['accumulated_lm_loss']
        accumulated_kl_loss = training_state['accumulated_kl_loss']
        micro_batch_count = training_state['micro_batch_count']
        if should_print and micro_batch_count > 0:
            print(f"  Note: Resuming mid-accumulation (micro_batch_count={micro_batch_count})")
    else:
        # Initialize as 0.0 (will be promoted to tensor on first accumulation)
        accumulated_loss = 0.0
        accumulated_lm_loss = 0.0
        accumulated_kl_loss = 0.0
        micro_batch_count = 0

    # Dynamic hyperparameters (updated by scheduler each step)
    current_temperature = args.temperature
    current_kl_weight = args.kl_weight

    # Router metrics aggregated over micro-batches within a single optimizer step
    # Why aggregate? Provides stable per-step statistics by averaging across all
    # micro-batches (e.g., 8 micro-batches per optimizer step with grad_accum=8).
    # This reduces noise and gives a better picture of routing behavior.
    #
    # Metrics tracked:
    # - layer_kl: KL divergence per layer (measures routing distribution shift from teacher)
    # - layer_l1: L1 distance between student/teacher routing probabilities
    # - student_entropy: Entropy of student routing (higher = more uniform, lower = more peaked)
    # - teacher_entropy: Entropy of teacher routing (reference baseline)
    step_router_metrics = {
        'layer_kl': [],
        'layer_l1': [],
        'student_entropy': [],
        'teacher_entropy': []
    }

    if global_step > 0:
        print(f"\nResuming training from step {global_step}...")
    else:
        print("\nStarting training...")

    print("="*80)
    if global_step == 0:
        print("\nNOTE: First batch may take several minutes (tokenizing, model warmup)")
    if args.log_every_n_steps > 1:
        print(f"Progress will be logged every {args.log_every_n_steps} steps (plus first 3 steps)")
    else:
        print("Progress will be logged every step")
    print("="*80)

    pbar = tqdm(total=max_steps, initial=global_step, desc="Training")

    # Validate shapes on first batch only (to save compute)
    validate_shapes = True

    # micro_batch_count already initialized above (either 0 or restored from checkpoint)
    total_batches_seen = 0  # Total batches processed since training start (for debugging/logging)

    # Epoch tracking (detects dataset exhaustion and restarts)
    # Restore from checkpoint if resuming
    if args.resume_from_checkpoint and 'current_epoch' in training_state:
        current_epoch = training_state['current_epoch']
        batches_in_current_epoch = training_state.get('batches_in_current_epoch', 0)
        if should_print and current_epoch > 0:
            print(f"  ✓ Resuming at epoch {current_epoch}, batch {batches_in_current_epoch} within epoch")
    else:
        current_epoch = 0
        batches_in_current_epoch = 0

    print("\n[Precomputed Data] Starting to iterate through precomputed teacher outputs...", flush=True)
    print(f"  Source: {args.precomputed_teacher}")
    print(f"  Available steps: {precomputed_loader.num_steps:,}")
    print(f"  Training steps: {max_steps:,} (gradient accumulation: {args.gradient_accumulation_steps})")

    # SIMPLIFIED: No dataloader, just iterate through precomputed files directly!
    # Each rank processes different steps in an interleaved pattern.
    rank = accelerator.process_index
    num_processes = accelerator.num_processes

    # Create iterator for this rank
    precomputed_iterator = precomputed_loader.iter_steps(rank=rank, num_processes=num_processes)

    # Track progress
    total_batches_seen = 0

    while global_step < max_steps:
        # Get next precomputed step for this rank
        try:
            precomputed_step, batch, teacher_outputs = next(precomputed_iterator)
        except StopIteration:
            # Exhausted precomputed data
            print(f"\n✗ Precomputed data exhausted at step {global_step}")
            print(f"  This should not happen - validation should have caught this!")
            raise RuntimeError(f"Precomputed data exhausted (needed {max_steps * accumulation_steps * num_processes} steps)")

        # -----------------------------------------------------------------------
        # DDP FIX: Use accelerator.accumulate context manager for DDP mode
        # This ensures gradient synchronization only happens on the final
        # micro-batch, not on every backward() call (8x reduction in sync ops)
        # -----------------------------------------------------------------------
        accumulate_context = accelerator.accumulate(student_model)
        with accumulate_context:
            # Progress logging for first few steps only
            should_log_detail = global_step < 3

            if should_log_detail:
                print(f"\n[Step {global_step}, Micro-batch {micro_batch_count+1}/{accumulation_steps}] Loading precomputed step {precomputed_step}...", flush=True)

            # OPTIONAL VERIFICATION: Verify sample integrity via hash
            # This catches data corruption or bugs in the precomputation/loading process
            if args.verify_sample_hashes:
                import hashlib
                # Compute hash of loaded input_ids
                input_ids_np = batch['input_ids'].cpu().numpy()
                hash_obj = hashlib.sha256()
                hash_obj.update(input_ids_np.tobytes())
                computed_hash = hash_obj.hexdigest()[:16]

                if computed_hash != teacher_outputs.input_ids_hash:
                    print(f"\n✗✗✗ SAMPLE INTEGRITY VERIFICATION FAILED ✗✗✗")
                    print(f"  Rank {rank}, Precomputed step {precomputed_step}")
                    print(f"  Expected hash: {teacher_outputs.input_ids_hash}")
                    print(f"  Computed hash: {computed_hash}")
                    print(f"  This indicates data corruption!")
                    raise RuntimeError(f"Sample integrity check failed at step {precomputed_step}")

                # Log verification success for first few steps
                if should_log_detail:
                    print(f"  ✓ Sample integrity verified (hash: {computed_hash})")

            if should_log_detail:
                ddp_info = f" (DDP rank {rank}/{num_processes})" if num_processes > 1 else ""
                print(f"  ✓ Using precomputed step {precomputed_step}{ddp_info}")
                print(f"    Sample already loaded (no dataloader replacement needed)")

            if should_log_detail:
                print(f"[Step {global_step}, Micro-batch {micro_batch_count+1}/{accumulation_steps}] Computing forward pass (student + precomputed teacher)...", flush=True)

            # Compute loss with KL (using dynamic temperature and kl_weight)
            loss, lm_loss, kl_loss, metrics = compute_loss_with_kl(
                student_model, None, batch,  # teacher_model=None for offline distillation
                kl_weight=current_kl_weight,
                temperature=current_temperature,
                expected_layers=num_layers,  # Auto-detected from model config
                validate_shapes=validate_shapes,
                teacher_outputs=teacher_outputs  # From precomputed data
            )

            if should_log_detail:
                # Detail logging: OK to sync here (only first 3 steps)
                print(f"[Step {global_step}, Micro-batch {micro_batch_count+1}/{accumulation_steps}] Forward complete. Loss: {loss.item():.4f}, LM: {lm_loss.item():.4f}, KL: {kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss:.4f}", flush=True)

            # Disable shape validation after first successful batch
            if validate_shapes and metrics['layer_kl']:
                validate_shapes = False
                print("✓ Router logits validation passed, disabling for subsequent batches")

            # Increment micro-batch counter
            micro_batch_count += 1
            total_batches_seen += 1

            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps

            if should_log_detail:
                print(f"[Step {global_step}, Micro-batch {micro_batch_count}/{accumulation_steps}] Running backward pass...", flush=True)

            # Backward pass (accelerator handles sync logic)
            accelerator.backward(loss)

            # PERFORMANCE FIX: Accumulate losses as tensors (avoid CUDA sync on every micro-batch)
            # Only convert to float when logging (once per optimizer step, not 8x per step)
            accumulated_loss += loss.detach()  # loss is already scaled by accumulation_steps
            accumulated_lm_loss += lm_loss / accumulation_steps  # lm_loss is a tensor now
            accumulated_kl_loss += kl_loss / accumulation_steps  # kl_loss is a tensor now

            # Aggregate router metrics across micro-batches for this optimizer step
            for key in step_router_metrics:
                step_router_metrics[key].extend(metrics[key])

            # -----------------------------------------------------------------------
            # DDP FIX: Use accelerator.sync_gradients for DDP mode
            # In non-DDP mode, fall back to manual micro_batch_count check
            # This flag is set by accelerator.accumulate on the final micro-batch
            # -----------------------------------------------------------------------
            should_step = accelerator.sync_gradients
            if should_step:
                    if should_log_detail:
                        print(f"\n[Step {global_step + 1}] Gradient accumulation complete ({micro_batch_count} micro-batches). Running optimizer step...", flush=True)

                    # Gradient clipping with detailed logging
                    # Use accelerator.clip_grad_norm_ for DDP (handles unscaling for mixed precision)
                    grad_norm_tensor = accelerator.clip_grad_norm_(
                        trainable_params,
                        args.gradient_clip_val
                    )

                    # Detach to avoid holding references to gradient buffers during long training runs
                    grad_norm = float(grad_norm_tensor.detach())

                    # Log gradient clipping info with step number for correlation
                    if should_log_detail or (grad_norm > args.gradient_clip_val * 1.1):  # Log if significant clipping
                        if grad_norm > args.gradient_clip_val:
                            clip_ratio = grad_norm / args.gradient_clip_val
                            # Format: GRADIENT_CLIP for easy parsing
                            clip_msg = f"GRADIENT_CLIP: step={global_step + 1}, norm={grad_norm:.4f}, threshold={args.gradient_clip_val}, ratio={clip_ratio:.2f}x"
                            print(clip_msg, flush=True)
                            tqdm.write(clip_msg)
                        else:
                            print(f"  [Step {global_step + 1}] Gradient norm: {grad_norm:.4f} (no clipping needed, threshold={args.gradient_clip_val})", flush=True)

                    # Optimizer step
                    optimizer.step()

                    # Update scheduler and get dynamic hyperparameters
                    if args.use_advanced_scheduler:
                        schedule_info = scheduler.step()
                        current_temperature = schedule_info['temperature']
                        current_kl_weight = schedule_info['kl_weight']
                    else:
                        scheduler.step()

                    optimizer.zero_grad()

                    global_step += 1
                    pbar.update(1)

                    # Calculate average metrics over all micro-batches in this optimizer step
                    # PERFORMANCE FIX: Metrics are now tensors - keep as tensors until after DDP reduction
                    if step_router_metrics['layer_kl']:
                        # Compute mean across micro-batches (keep as tensor for DDP reduction)
                        avg_layer_kl_tensor = torch.stack(step_router_metrics['layer_kl']).mean()
                        avg_l1_tensor = torch.stack(step_router_metrics['layer_l1']).mean()
                        avg_s_entropy_tensor = torch.stack(step_router_metrics['student_entropy']).mean()
                        avg_t_entropy_tensor = torch.stack(step_router_metrics['teacher_entropy']).mean()

                        # DDP FIX: Reduce router metrics across all ranks to get global average
                        if accelerator.num_processes > 1:
                            avg_layer_kl_tensor = accelerator.reduce(avg_layer_kl_tensor, reduction='mean')
                            avg_l1_tensor = accelerator.reduce(avg_l1_tensor, reduction='mean')
                            avg_s_entropy_tensor = accelerator.reduce(avg_s_entropy_tensor, reduction='mean')
                            avg_t_entropy_tensor = accelerator.reduce(avg_t_entropy_tensor, reduction='mean')

                        # Convert to float after DDP reduction
                        avg_layer_kl = float(avg_layer_kl_tensor.item())
                        avg_l1 = float(avg_l1_tensor.item())
                        avg_s_entropy = float(avg_s_entropy_tensor.item())
                        avg_t_entropy = float(avg_t_entropy_tensor.item())

                        # Calculate routing entropy deviation (student - teacher)
                        entropy_deviation = avg_s_entropy - avg_t_entropy

                        # Log warnings if entropy deviation crosses thresholds
                        if abs(entropy_deviation) > 0.1:
                            print(f"  ⚠️  ROUTING_DRIFT: step={global_step}, deviation={entropy_deviation:.4f} (|dev| > 0.1)", flush=True)
                            tqdm.write(f"  ⚠️  ROUTING_DRIFT: step={global_step}, deviation={entropy_deviation:.4f}")
                        elif abs(entropy_deviation) > 0.05:
                            # Log moderate drift only on detail steps or every 100 steps
                            if should_log_detail or global_step % 100 == 0:
                                print(f"  ⚠  Moderate routing drift: deviation={entropy_deviation:.4f}", flush=True)
                    else:
                        # No router metrics collected (e.g., model did not return router_logits)
                        avg_layer_kl = 0.0
                        avg_l1 = 0.0
                        avg_s_entropy = 0.0
                        avg_t_entropy = 0.0
                        entropy_deviation = 0.0

                    # PERFORMANCE OPTIMIZATION: Only save metrics at log intervals
                    # Reduces CUDA syncs from 151/step to 11/step-at-log-interval
                    # For default log_every_n_steps=10: 15.1M syncs → 110K syncs (137x reduction, ~125 min saved)
                    should_save_metrics = (global_step % args.log_every_n_steps == 0) or \
                                         (not args.disable_sfa and args.sfa_merge_interval > 0 and global_step % args.sfa_merge_interval == 0)

                    if should_save_metrics:
                        # DDP FIX: Reduce losses across all ranks to get global average
                        # Without this, we only log rank 0's local loss (misleading if data distribution varies)
                        if accelerator.num_processes > 1:
                            accumulated_loss = accelerator.reduce(accumulated_loss, reduction='mean')
                            accumulated_lm_loss = accelerator.reduce(accumulated_lm_loss, reduction='mean')
                            accumulated_kl_loss = accelerator.reduce(accumulated_kl_loss, reduction='mean')
                            # Note: Router metrics are already averaged per-rank, reduction happens below

                        # Convert accumulated tensors to floats (7 syncs)
                        loss_float = float(accumulated_loss.item()) if isinstance(accumulated_loss, torch.Tensor) else accumulated_loss
                        lm_loss_float = float(accumulated_lm_loss.item()) if isinstance(accumulated_lm_loss, torch.Tensor) else accumulated_lm_loss
                        kl_loss_float = float(accumulated_kl_loss.item()) if isinstance(accumulated_kl_loss, torch.Tensor) else accumulated_kl_loss

                        # Log to console (only at log intervals, not at SFA-only saves)
                        if global_step % args.log_every_n_steps == 0:
                            log_msg = (
                                f"Step {global_step}, epoch={current_epoch}: "
                                f"loss={loss_float:.4f}, "
                                f"lm={lm_loss_float:.4f}, "
                                f"kl={kl_loss_float:.4f}, "
                                f"l1_dist={avg_l1:.4f}, "
                                f"s_entropy={avg_s_entropy:.4f}, "
                                f"t_entropy={avg_t_entropy:.4f}, "
                                f"grad_norm={grad_norm:.4f}, "
                                f"lr={scheduler.get_last_lr()[0]:.2e}, "
                                f"temp={current_temperature:.2f}, "
                                f"kl_weight={current_kl_weight:.3f}"
                            )
                            tqdm.write(log_msg)
                            print(log_msg, flush=True)

                        # PERFORMANCE FIX: Batch convert tensor lists (4 syncs instead of 144)
                        # OLD: [float(t.item()) for t in tensors] → 36 syncs/metric × 4 metrics = 144 syncs
                        # NEW: torch.stack(tensors).cpu().tolist() → 1 sync/metric × 4 metrics = 4 syncs
                        layer_kl_floats = torch.stack(step_router_metrics['layer_kl']).cpu().tolist() if step_router_metrics['layer_kl'] else []
                        layer_l1_floats = torch.stack(step_router_metrics['layer_l1']).cpu().tolist() if step_router_metrics['layer_l1'] else []
                        student_entropy_floats = torch.stack(step_router_metrics['student_entropy']).cpu().tolist() if step_router_metrics['student_entropy'] else []
                        teacher_entropy_floats = torch.stack(step_router_metrics['teacher_entropy']).cpu().tolist() if step_router_metrics['teacher_entropy'] else []

                        # Store detailed metrics for JSON serialization
                        step_metrics = {
                            'step': global_step,
                            'epoch': current_epoch,  # Track which epoch (dataset pass) we're on
                            'loss': loss_float,
                            'lm_loss': lm_loss_float,
                            'kl_loss': kl_loss_float,
                            'avg_layer_kl': avg_layer_kl,
                            'layer_kl': layer_kl_floats,
                            'avg_l1_distance': avg_l1,
                            'layer_l1': layer_l1_floats,
                            'avg_student_entropy': avg_s_entropy,
                            'avg_teacher_entropy': avg_t_entropy,
                            'entropy_deviation': entropy_deviation,
                            'gradient_norm': float(grad_norm),
                            'gradient_clipped': bool(grad_norm > args.gradient_clip_val),
                            'learning_rate': scheduler.get_last_lr()[0],
                            'temperature': current_temperature,
                            'kl_weight': current_kl_weight,
                            'sfa_merged': False,  # Will be set to True if merge happens
                        }
                    else:
                        # Not saving metrics this step - set to None to skip append later
                        step_metrics = None

                    # SFA: Periodic merge with anchor checkpoint (ROLLING APPROACH)
                    # NOTE: Merge happens BEFORE checkpoint save, so saved checkpoints contain merged weights
                    # STRATEGY: Skip first checkpoint, then merge with previous checkpoint (not original anchor)
                    #   - Step 250: Save checkpoint, NO merge (let training progress freely)
                    #   - Step 500: Merge with checkpoint-250, then save
                    #   - Step 750: Merge with checkpoint-500, then save
                    #   This allows gradual drift while preventing catastrophic forgetting
                    if not args.disable_sfa and args.sfa_merge_interval > 0 and global_step % args.sfa_merge_interval == 0:
                        # Skip merge on first merge point (allow free training initially)
                        if global_step == args.sfa_merge_interval:
                            print(f"\n{'='*80}")
                            print(f"SFA: Skipping merge at Step {global_step} (first merge point)")
                            print(f"{'='*80}")
                            print(f"  Allowing free training for initial {args.sfa_merge_interval} steps")
                            print(f"  Next merge at step {global_step + args.sfa_merge_interval} will use a previous checkpoint as anchor")
                            print(f"{'='*80}\n")
                            if step_metrics is not None:
                                step_metrics['sfa_merged'] = False
                                step_metrics['sfa_skipped_first'] = True
                        else:
                            # DDP-SAFE SFA MERGE: Only rank 0 merges, then broadcasts to prevent FPU drift
                            # CPU floating point ops are non-deterministic across processes → models diverge
                            # Solution: Rank 0 merges, broadcasts merged state_dict to all ranks
                            prev_checkpoint_step = global_step - args.checkpoint_interval
                            prev_checkpoint_dir = output_dir / f"checkpoint-{prev_checkpoint_step}"

                            if should_print:
                                print(f"\n{'='*80}")
                                print(f"SFA MERGE at Step {global_step} (ROLLING ANCHOR - OPTIMIZED)")
                                print(f"{'='*80}")
                                print(f"Loading ONLY trainable weights from checkpoint-{prev_checkpoint_step}")
                                print(f"  (Memory-efficient: ~900MB instead of 60GB)")

                            # RANK 0 ONLY: Load and merge
                            merged_state_dict = None
                            merge_stats = None
                            if should_print:
                                # Load only trainable parameters from previous checkpoint (on CPU)
                                try:
                                    anchor_state_dict = load_trainable_weights_from_checkpoint(
                                        prev_checkpoint_dir,
                                        trainable_param_names
                                    )
                                except FileNotFoundError as e:
                                    print(f"\n  ✗ ERROR: Previous checkpoint not found: {prev_checkpoint_dir}")
                                    print(f"  Cannot perform SFA merge. Skipping merge at step {global_step}.")
                                    print(f"  Error details: {e}")
                                    print(f"  Training will continue without merge at this step.\n")
                                    if step_metrics is not None:
                                        step_metrics['sfa_merged'] = False
                                        step_metrics['sfa_error'] = str(e)
                                        all_metrics.append(step_metrics)
                                    # Reset losses and counters, then continue training
                                    accumulated_loss = 0.0
                                    accumulated_lm_loss = 0.0
                                    accumulated_kl_loss = 0.0
                                    micro_batch_count = 0
                                    step_router_metrics = {
                                        'layer_kl': [],
                                        'layer_l1': [],
                                        'student_entropy': [],
                                        'teacher_entropy': []
                                    }
                                    continue  # Skip rest of merge logic

                                print(f"  Merging current weights with checkpoint-{prev_checkpoint_step}...")
                                print(f"  Alpha: {args.sfa_alpha}")
                                print(f"  Formula: merged = {args.sfa_alpha} * checkpoint-{prev_checkpoint_step} + {1-args.sfa_alpha} * current")
                                print(f"  Strategy: Merge on CPU, then broadcast to all ranks")

                                # MEMORY OPTIMIZATION: merge_with_anchor now returns merged_state_dict directly
                                # This avoids calling state_dict() after the merge (which would waste 60GB)
                                merge_stats, merged_state_dict = merge_with_anchor(student_model, anchor_state_dict, alpha=args.sfa_alpha)

                                print(f"\nMerge complete:")
                                print(f"  Trainable parameters merged: {merge_stats['trainable_params_merged']:,}")
                                print(f"  Max change in trainable params: {merge_stats['max_change']:.6f}")
                                print(f"  Mean change in trainable params: {merge_stats['mean_change']:.6f}")
                                print(f"  Broadcasting {len(merged_state_dict)} trainable params to all ranks...")

                            # DDP SYNC: Broadcast merged state_dict from rank 0 to all ranks
                            if accelerator.num_processes > 1:
                                # Use broadcast_object_list for state_dict (works for any picklable object)
                                from accelerate.utils import broadcast_object_list
                                merged_state_list = [merged_state_dict]  # Wrap in list for broadcast
                                broadcast_object_list(merged_state_list, from_process=0)
                                merged_state_dict = merged_state_list[0]

                                # MEMORY OPTIMIZATION: All non-rank-0 processes update in-place
                                # Rank 0 already updated during merge, non-rank-0 update here
                                if not should_print:  # Non-rank-0 processes only
                                    model_to_load = accelerator.unwrap_model(student_model)
                                    for name, param in model_to_load.named_parameters():
                                        if name in merged_state_dict:
                                            # IN-PLACE UPDATE: Directly copy to parameter data
                                            param.data.copy_(merged_state_dict[name])

                                if should_print:
                                    print(f"  ✓ All ranks synchronized with rank 0 merged weights")
                                    print(f"{'='*80}\n")

                            # Add merge stats to metrics (if we saved metrics this step)
                            if step_metrics is not None and should_print:
                                step_metrics['sfa_merged'] = True
                                step_metrics['sfa_merge_stats'] = merge_stats
                                step_metrics['sfa_anchor_checkpoint'] = prev_checkpoint_step

                    # Only append metrics if we saved them this step
                    if step_metrics is not None:
                        all_metrics.append(step_metrics)

                    # Reset accumulated losses and micro-batch counter
                    accumulated_loss = 0.0
                    accumulated_lm_loss = 0.0
                    accumulated_kl_loss = 0.0
                    micro_batch_count = 0  # Reset for next accumulation window

                    # Save checkpoint (only on main process in DDP)
                    if global_step % args.checkpoint_interval == 0:
                        if should_print:  # Only main process saves
                            checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                            checkpoint_dir.mkdir(exist_ok=True)

                            print(f"\nSaving checkpoint at step {global_step}...")

                            # Unwrap model from DDP if using Accelerate
                            model_to_save = accelerator.unwrap_model(student_model)

                            # Remove PEFT flag temporarily for saving
                            if hasattr(model_to_save, '_hf_peft_config_loaded'):
                                delattr(model_to_save, '_hf_peft_config_loaded')

                            model_to_save.save_pretrained(checkpoint_dir)
                            tokenizer.save_pretrained(checkpoint_dir)

                            # Restore PEFT flag
                            model_to_save._hf_peft_config_loaded = True

                            # Save ONLY trainable weights as separate safetensors file for fast SFA merging (if SFA enabled)
                            if not args.disable_sfa:
                                print(f"  Saving trainable weights separately for fast SFA merging...")
                                save_trainable_weights_safetensors(model_to_save, checkpoint_dir)
                            else:
                                print(f"  Skipping trainable_weights.safetensors (SFA disabled)")

                            # Save training state for resumption
                            print(f"  Saving training state (optimizer, scheduler, random state)...")
                            # PERFORMANCE FIX: Convert accumulated losses to floats if they're tensors
                            acc_loss_save = float(accumulated_loss.item()) if isinstance(accumulated_loss, torch.Tensor) else accumulated_loss
                            acc_lm_loss_save = float(accumulated_lm_loss.item()) if isinstance(accumulated_lm_loss, torch.Tensor) else accumulated_lm_loss
                            acc_kl_loss_save = float(accumulated_kl_loss.item()) if isinstance(accumulated_kl_loss, torch.Tensor) else accumulated_kl_loss

                            training_state = {
                                'global_step': global_step,
                                'current_epoch': current_epoch,  # Track dataset pass for resumption
                                'batches_in_current_epoch': batches_in_current_epoch,  # Track position within epoch
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                                'random_state': {
                                    'torch': torch.get_rng_state(),
                                    'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                                    'numpy': np.random.get_state(),
                                    'python': random.getstate(),
                                },
                                'args': vars(args),  # Save all training arguments
                                'accumulated_loss': acc_loss_save,
                                'accumulated_lm_loss': acc_lm_loss_save,
                                'accumulated_kl_loss': acc_kl_loss_save,
                                'micro_batch_count': micro_batch_count,
                            }
                            torch.save(training_state, checkpoint_dir / "training_state.pt")

                            # Save metrics
                            with open(checkpoint_dir / "metrics.json", "w") as f:
                                json.dump(all_metrics, f, indent=2)

                            print(f"Checkpoint saved to {checkpoint_dir}")

                        # Synchronize all processes after checkpoint (ensure all see the same step)
                        accelerator.wait_for_everyone()

                        # Analyze router divergence at checkpoints
                        if step_router_metrics['layer_kl']:
                            print("\nRouter Divergence Analysis:")
                            print(f"  Average KL across layers: {avg_layer_kl:.6f}")
                            print(f"  Min layer KL: {min(step_router_metrics['layer_kl']):.6f}")
                            print(f"  Max layer KL: {max(step_router_metrics['layer_kl']):.6f}")
                            print(f"  Average L1 distance: {avg_l1:.4f}")
                            print(f"  Student entropy: {avg_s_entropy:.4f}")
                            print(f"  Teacher entropy: {avg_t_entropy:.4f}")

                            # Check if routers are diverging
                            if avg_layer_kl > 0.001:
                                print("  ✓ Routers are diverging from teacher (good)")
                            else:
                                print("  ⚠ Routers still very similar to teacher")

                            if avg_l1 > 0.01:
                                print("  ✓ Significant L1 distance between distributions")
                            else:
                                print("  ⚠ Distributions still very similar")
                        else:
                            print("\nRouter Divergence Analysis:")
                            print("  No router metrics collected for this step.")

                    # Reset aggregated router metrics for the next optimizer step
                    step_router_metrics = {
                        'layer_kl': [],
                        'layer_l1': [],
                        'student_entropy': [],
                        'teacher_entropy': []
                    }

                    if global_step >= max_steps:
                        break

        # Note: Dataset exhaustion handling removed - we iterate through precomputed files directly
        # If we exit the loop before reaching max_steps, it means precomputed data was exhausted
        # (which should have been caught during validation)

    pbar.close()

    # Save final model (only on main process in DDP)
    if should_print:
        print("\nSaving final model...")
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)

        # Unwrap model from DDP if using Accelerate
        model_to_save = accelerator.unwrap_model(student_model)

        # Remove PEFT flag for final save
        if hasattr(model_to_save, '_hf_peft_config_loaded'):
            delattr(model_to_save, '_hf_peft_config_loaded')

        model_to_save.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

        # Save trainable weights separately (if SFA enabled)
        if not args.disable_sfa:
            print(f"  Saving trainable weights separately...")
            save_trainable_weights_safetensors(model_to_save, final_dir)
        else:
            print(f"  Skipping trainable_weights.safetensors (SFA disabled)")

        # Save all metrics
        with open(final_dir / "all_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

    # Synchronize before final summary
    accelerator.wait_for_everyone()

    if should_print:
        print(f"\nTraining complete! Final model saved to {final_dir}")

    # SFA Summary (only on main process)
    if should_print:
        num_merges = sum(1 for m in all_metrics if m.get('sfa_merged', False))
        if num_merges > 0:
            print(f"\n{'='*80}")
            print("SFA SUMMARY")
            print(f"{'='*80}")
            print(f"Total merges performed: {num_merges}")
            print(f"Merge interval: {args.sfa_merge_interval} steps")
            print(f"Alpha: {args.sfa_alpha}")
            merge_steps = [m['step'] for m in all_metrics if m.get('sfa_merged', False)]
            print(f"Merge occurred at steps: {merge_steps}")
            print(f"{'='*80}")

    # Final analysis (only on main process)
    if should_print and all_metrics:
        final_metrics = all_metrics[-1]
        print("\n" + "="*80)
        print("Final Training Metrics:")
        print("="*80)
        print(f"Total Loss: {final_metrics['loss']:.4f}")
        print(f"LM Loss: {final_metrics['lm_loss']:.4f}")
        print(f"KL Loss: {final_metrics['kl_loss']:.4f}")
        print(f"Average Layer KL: {final_metrics['avg_layer_kl']:.6f}")
        print(f"L1 Distance: {final_metrics['avg_l1_distance']:.4f}")
        print(f"Student Entropy: {final_metrics['avg_student_entropy']:.4f}")
        print(f"Teacher Entropy: {final_metrics['avg_teacher_entropy']:.4f}")
        print(f"Final Gradient Norm: {final_metrics['gradient_norm']:.4f}")

        # Gradient statistics
        grad_norms = [m['gradient_norm'] for m in all_metrics]
        clipped_steps = sum(1 for m in all_metrics if m.get('gradient_clipped', False))
        print(f"\nGradient Statistics:")
        print(f"  Average norm: {np.mean(grad_norms):.4f}")
        print(f"  Min norm: {np.min(grad_norms):.4f}")
        print(f"  Max norm: {np.max(grad_norms):.4f}")
        print(f"  Std dev: {np.std(grad_norms):.4f}")
        print(f"  Steps clipped: {clipped_steps}/{len(all_metrics)} ({100*clipped_steps/len(all_metrics):.1f}%)")

        # Diagnosis
        print("\n" + "="*80)
        print("Training Diagnosis:")
        print("="*80)

        if final_metrics['kl_loss'] > 0:
            print("✓ KL loss is positive (mathematically valid)")
        else:
            print("✗ KL loss is negative or zero (indicates numerical issues)")

        if final_metrics['avg_layer_kl'] > 0.01:
            print("✓ Strong KL divergence - routers have learned different patterns")
        elif final_metrics['avg_layer_kl'] > 0.001:
            print("✓ Moderate KL divergence - routers are starting to diverge")
        else:
            print("⚠ Weak KL divergence - consider more training steps or higher LR")

        if final_metrics['avg_l1_distance'] > 0.1:
            print("✓ Significant routing distribution changes")
        elif final_metrics['avg_l1_distance'] > 0.01:
            print("✓ Moderate routing distribution changes")
        else:
            print("⚠ Minimal routing distribution changes")

        # Routing entropy deviation analysis
        entropy_deviations = [m.get('entropy_deviation', 0.0) for m in all_metrics if 'entropy_deviation' in m]
        if entropy_deviations:
            mean_dev = np.mean(entropy_deviations)
            std_dev = np.std(entropy_deviations)
            max_abs_dev = max(abs(d) for d in entropy_deviations)

            print(f"\nRouting Entropy Deviation (Student - Teacher):")
            print(f"  Mean deviation: {mean_dev:+.4f} bits")
            print(f"  Std deviation: {std_dev:.4f} bits")
            print(f"  Max |deviation|: {max_abs_dev:.4f} bits")

            # Interpretation
            if abs(mean_dev) < 0.01 and std_dev < 0.02:
                print(f"  ✓ Excellent alignment - minimal routing impact")
            elif abs(mean_dev) < 0.05 and std_dev < 0.05:
                print(f"  ✓ Good alignment - controlled routing drift")
            elif std_dev < 0.1:
                print(f"  ⚠ Moderate drift - shared expert affecting routing")
            else:
                print(f"  ⚠ Significant drift - major routing changes")

            # Count threshold crossings
            moderate_drifts = sum(1 for d in entropy_deviations if abs(d) > 0.05)
            severe_drifts = sum(1 for d in entropy_deviations if abs(d) > 0.1)
            if severe_drifts > 0:
                print(f"  Warning: {severe_drifts} steps with |deviation| > 0.1")
            if moderate_drifts > 0:
                print(f"  Note: {moderate_drifts} steps with |deviation| > 0.05")


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS Offline Distillation Training with DDP + SFA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OFFLINE DISTILLATION with DDP:
Uses precomputed teacher outputs (teacher model NOT loaded - saves ~80GB VRAM).
Distributed Data Parallel training across multiple GPUs with Accelerate.

SFA (Sequential Fine-tuning with Averaging) mitigates catastrophic forgetting using
a ROLLING ANCHOR approach: merge with previous checkpoint (not initial checkpoint).

Example usage:
  # Step 1: Precompute teacher outputs (run once)
  python precompute_teacher.py \\
      --teacher-model /path/to/gpt-oss-120b \\
      --output teacher_outputs/ \\
      --num-steps 100000

  # Step 2: Train with DDP
  accelerate launch --num_processes=2 train_offline.py \\
      --precomputed-teacher teacher_outputs/ \\
      --student-model /path/to/gpt-oss-120b-nexus \\
      --max-steps 2000 \\
      --checkpoint-interval 500 \\
      --sfa-merge-interval 500 \\
      --sfa-alpha 0.25 \\
      --output-dir outputs/stage0_sfa

  Note: sfa-merge-interval will be auto-synced to checkpoint-interval

Alpha (anchor weight) guidelines:
  0.1-0.2: Mostly keep new learning, light regularization
  0.25:    Balanced (recommended for T=4.0 → T=1.0 fine-tuning)
  0.3-0.4: Conservative, stronger retention of previous checkpoint
        """
    )

    # Model arguments
    parser.add_argument("--student-model", type=str, required=True,
                       help="Path to student model with shared expert")
    parser.add_argument("--precomputed-teacher", type=str, required=True,
                       help="Path to directory with precomputed teacher outputs (REQUIRED for offline distillation)")

    # Training arguments
    # Note: Training steps automatically determined from precomputed data
    # max_steps = num_precomputed_steps // (gradient_accumulation_steps × num_processes)
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size (should match precomputed data)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0,
                       help="Gradient clipping value (default: 1.0). Set to large value (e.g., 1000) to effectively disable.")

    # Performance arguments
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory (enables longer sequences)")
    # Note: DDP is always enabled for this script (launched via accelerate launch)

    # SFA arguments
    parser.add_argument("--sfa-merge-interval", type=int, default=250,
                       help="Merge interval (default: 250). Will be auto-synced to checkpoint-interval for "
                            "rolling anchor approach. Set to 0 to disable SFA *merging* while still using "
                            "constant LR schedule and saving trainable weights. Use --disable-sfa to fully "
                            "disable SFA behavior (merging + constant LR + trainable weights).")
    parser.add_argument("--sfa-alpha", type=float, default=0.25,
                       help="Weight for previous checkpoint in merge (default: 0.25). "
                            "merged = alpha*prev_checkpoint + (1-alpha)*current. "
                            "Higher = more conservative, retains more from previous checkpoint.")
    parser.add_argument("--disable-sfa", action="store_true", default=False,
                       help="Fully disable SFA mode (default: False). When enabled, disables SFA merging, "
                            "trainable weights saving, and switches from constant LR to standard LR decay. "
                            "Use this for baseline training runs.")

    # KL distillation arguments (IMPROVED DEFAULTS)
    parser.add_argument("--kl-weight", type=float, default=1.0,
                       help="Weight for KL loss (default: 1.0 for equal importance)")
    parser.add_argument("--temperature", type=float, default=4.0,
                       help="Temperature for KL scaling (default: 4.0)")

    # Advanced scheduler arguments (DeepSeek-V3 style)
    parser.add_argument("--use-advanced-scheduler", action="store_true",
                       help="Use advanced DeepSeek-V3 style scheduler with warmup→stable→decay, "
                            "router bias scheduling, temperature annealing, and KL weight warmup")
    parser.add_argument("--warmup-ratio", type=float, default=0.5,
                       help="Fraction of steps for LR warmup (default: 0.5 = 50%% - longer warmup for stability)")
    parser.add_argument("--stable-ratio", type=float, default=0.2,
                       help="Fraction of steps at peak LR (default: 0.2 = 20%%)")
    parser.add_argument("--final-lr-ratio", type=float, default=0.1,
                       help="Final LR as fraction of peak LR (default: 0.1 = 10%%, NOT zero)")
    parser.add_argument("--bias-lr-multiplier", type=float, default=0.001,
                       help="Router bias LR multiplier (default: 0.001 = 0.1%% of main LR)")
    parser.add_argument("--bias-freeze-ratio", type=float, default=None,
                       help="Freeze router biases after this fraction of training (default: None = freeze at end of warmup). "
                            "For pretrained models, freezing early (at warmup end) prevents destabilization.")
    parser.add_argument("--freeze-router", action="store_true",
                       help="Freeze ALL router parameters (weights + biases). Train only shared expert. "
                            "Recommended for PCA-initialized models where router is already well-trained. "
                            "Reduces trainable params from ~909M to ~896M and simplifies optimization.")
    parser.add_argument("--final-temperature", type=float, default=2.0,
                       help="Final temperature for KL distillation (default: 2.0, anneals from --temperature)")
    parser.add_argument("--temperature-anneal-start", type=float, default=0.3,
                       help="Start temperature annealing after this fraction (default: 0.3 = 30%%)")
    parser.add_argument("--kl-weight-warmup", action="store_true",
                       help="Gradually increase KL weight from 0 to 1 during warmup")

    # Note: Device placement is handled by Accelerate in DDP mode

    # Data arguments - REMOVED (no longer needed for offline training)
    # All data comes from precomputed teacher outputs, which already contain:
    # - input_ids (samples)
    # - attention_mask
    # - teacher logits and router logits
    # Batch size and sequence length are read from precomputed metadata.

    # Optional verification
    parser.add_argument("--verify-sample-hashes", action="store_true",
                       help="Verify input_ids hashes for each batch (optional, for debugging). "
                            "Adds ~0.1ms per batch to detect data corruption.")

    # Note: num_layers is automatically detected from student model config
    # Supports both GPT-OSS 20B (24 layers) and 120B (36 layers)

    # Reproducibility arguments
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (default: None = no seeding)")
    parser.add_argument("--deterministic", action="store_true",
                       help="Enable full deterministic training (slower but fully reproducible)")

    # Output arguments
    parser.add_argument("--output-dir", type=str,
                       default="outputs/stage0_improved",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                       help="Resume training from checkpoint directory (loads optimizer, scheduler, random state)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log-every-n-steps", type=int, default=10,
                       help="Log training metrics every N steps (default: 10)")

    args = parser.parse_args()

    # Validate required arguments for offline distillation
    if not args.precomputed_teacher:
        raise ValueError(
            "ERROR: --precomputed-teacher is required for offline distillation training.\n"
            "\n"
            "This script requires precomputed teacher outputs. To create them:\n"
            "  python scripts/gpt_oss/precompute_teacher.py \\\n"
            "      --teacher-model /path/to/teacher \\\n"
            "      --output teacher_outputs/ \\\n"
            "      --num-steps 100000\n"
            "\n"
            "Then run this script:\n"
            "  accelerate launch --num_processes=2 train_offline.py \\\n"
            "      --precomputed-teacher teacher_outputs/ \\\n"
            "      --student-model /path/to/student\n"
        )

    # Run training
    train(args)


if __name__ == "__main__":
    main()