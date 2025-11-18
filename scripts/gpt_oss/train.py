#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
GPT-OSS Online Distillation Training

Trains student model with live teacher model (teacher on GPU 0, student on GPU 1).
No DDP - single process with two GPUs. Teacher forward pass computed inline.

Implements Sequential Fine-tuning with Averaging (SFA) based on:
"Soup to go: mitigating forgetting during continual learning with model averaging"
https://arxiv.org/abs/2501.05559

Key features:
- Online distillation: Teacher model loaded and runs forward pass during training
- Two-GPU setup: Teacher on GPU 0, student on GPU 1 (no DDP)
- SFA: Rolling anchor merging for continual learning
- Simple gradient accumulation: No Accelerate overhead

================================================================================
HARDWARE REQUIREMENTS
================================================================================

- Teacher model: 1x RTX Pro 6000 Blackwell (~98GB VRAM) - GPU 0
- Student model: 1x RTX Pro 6000 Blackwell (~98GB VRAM) - GPU 1
- Total: 2x GPUs required (each model on dedicated GPU)

Memory per GPU:
- Model weights: ~70GB (MXFP4 quantized routed experts + BF16 shared expert)
- Training overhead: ~5-10GB (gradients + optimizer states)
- Peak: ~75-80GB per GPU (fits on 98GB cards)

Why separate GPUs?
- Each 120B model needs ~75-80GB VRAM
- Running both on same GPU would cause OOM
- Teacher kept in memory for inline forward pass

================================================================================
USAGE
================================================================================

Basic training:
    python train.py \\
        --teacher-model /path/to/gpt-oss-120b \\
        --student-model /path/to/gpt-oss-120b-nexus \\
        --teacher-device cuda:0 \\
        --student-device cuda:1 \\
        --output-dir outputs/ \\
        --max-steps 10000

For DDP training, use train_offline.py instead with precomputed teacher outputs.

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

# Local imports - shared modules
from dataset import create_dataloader
from distillation import compute_loss_with_kl
from sfa import (
    load_trainable_weights_from_checkpoint,
    merge_with_anchor,
    save_trainable_weights_safetensors
)
from model_utils import (
    load_teacher_model,
    load_student_model,
    prepare_model_for_stage0,
    validate_device_config,
    set_deterministic_training
)
from scheduler import GPTOSSTrainingScheduler, create_optimizer_with_bias_groups



def train(args):
    """
    Main training loop with SFA (Sequential Fine-tuning with Averaging).

    This function orchestrates the complete training process for Stage-0, which trains only the
    routers and shared expert while keeping MXFP4-quantized routed experts frozen.

    Key Features:
    - KL distillation from teacher to student model (temperature-scaled)
    - Rolling anchor SFA: periodically merge with previous checkpoint (optional)
    - Memory-efficient: trains only ~909M params out of 120B total (~0.76%)
    - Single-GPU-per-model: MXFP4 quantization + gradient checkpointing allows
      each 120B model to fit on one workstation GPU (default: teacher=cuda:0, student=cuda:1)

    Training Flow:
    1. Setup: Load models, tokenizer, optimizer, scheduler
    2. Training loop: Gradient accumulation with KL distillation
    3. Checkpointing: Save model + metrics every N steps
    4. SFA merging: Periodically merge with previous checkpoint (if enabled)
    5. Final save: Save final model and comprehensive metrics

    Args:
        args: Parsed command-line arguments containing all hyperparameters.
              Key args: teacher_model, student_model, max_steps, learning_rate,
              sfa_merge_interval, sfa_alpha, gradient_checkpointing, etc.

    Memory Usage (per GPT-OSS-120B model on RTX Pro 6000 Blackwell):
        - Model weights: ~70GB (MXFP4 quantized)
        - Training overhead: ~5-10GB (gradients + optimizer states)
        - Peak: ~75-80GB (fits comfortably on 98GB card with ~20GB headroom)
        - Typical setup: 2x RTX Pro 6000 Blackwell GPUs (one for teacher, one for student)

    Returns:
        None (saves checkpoints and metrics to disk)
    """

    print("="*80)
    print("ONLINE Distillation Training with SFA")
    print("="*80)

    # No Accelerator/DDP for online distillation (simple single-process training)
    should_print = True  # Always print in single-process mode

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
    print(f"Max Steps: {args.max_steps}")
    print(f"Checkpoint Interval: {args.checkpoint_interval}")
    print(f"Expected Layers: [Auto-detected from model]")
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
        config['use_ddp'] = False  # Online mode doesn't use DDP
        config['online_distillation'] = True

        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    # Validate and parse device configuration (uses model_utils.validate_device_config)
    (teacher_backend, teacher_device_id), (student_backend, student_device_id) = validate_device_config(
        args.teacher_device, args.student_device, precomputed=False
    )

    # Load teacher model for online distillation (uses model_utils.load_teacher_model)
    teacher_model = load_teacher_model(args.teacher_model, teacher_device_id)

    # Auto-detect model configuration
    from transformers import AutoConfig
    student_config = AutoConfig.from_pretrained(args.student_model, trust_remote_code=True)
    num_layers = student_config.num_hidden_layers

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

    # Load student model (uses model_utils.load_student_model)
    student_model = load_student_model(
        args.student_model,
        student_device_id,
        use_ddp=False,  # Online mode doesn't use DDP
        local_rank=None
    )

    # Validate that teacher and student are different models
    # (same path is unusual but allowed - typically student has modified architecture)
    if args.teacher_model == args.student_model:
        print("\n" + "="*80)
        print("WARNING: Teacher and student models use the same path!")
        print("="*80)
        print(f"  Path: {args.teacher_model}")
        print("  This is unusual - typically the student has a modified architecture")
        print("  (e.g., with added shared expert) while teacher is the original model.")
        print("  Training will work but may not be effective if models are identical.")
        print("="*80 + "\n")

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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create data
    # Build local dataset paths dict if provided
    local_paths = None
    if args.local_nemotron_code and args.local_nemotron_math and args.local_nemotron_tool:
        local_paths = {
            'code': args.local_nemotron_code,
            'math': args.local_nemotron_math,
            'tool': args.local_nemotron_tool
        }

    train_dataloader = create_dataloader(
        tokenizer,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seq_len=args.seq_len,
        num_workers=0,  # Keep 0 for compatibility with streaming
        dataset_name=args.dataset,
        local_dataset_paths=local_paths
    )

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
    num_training_steps = args.max_steps

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

    # No DDP preparation needed for online distillation (single process, no data parallelism)

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
            print(f"  Remaining steps: {args.max_steps - global_step}")

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

    pbar = tqdm(total=args.max_steps, initial=global_step, desc="Training")

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

    print("\n[DataLoader] Starting to iterate through dataset...", flush=True)
    if args.max_samples is not None:
        batches_needed = args.max_steps * args.gradient_accumulation_steps
        print(f"  Dataset samples: {args.max_samples:,}")
        print(f"  Batches needed: {batches_needed:,}")
        if batches_needed > args.max_samples:
            epochs_expected = batches_needed / args.max_samples
            print(f"  ⚠️  Dataset will repeat {epochs_expected:.1f}x (train for {epochs_expected:.0f} epochs)")
            print(f"     First repeat at step ~{args.max_samples // args.gradient_accumulation_steps:,}")
        else:
            print(f"  ✓ Sufficient data (no repetition expected)")

    while global_step < args.max_steps:
        for batch in train_dataloader:
            batches_in_current_epoch += 1
            if global_step >= args.max_steps:
                break

            # -----------------------------------------------------------------------
            # DDP FIX: Use accelerator.accumulate context manager for DDP mode
            # In non-DDP mode (accelerator=None), use nullcontext (no-op)
            # This ensures gradient synchronization only happens on the final
            # micro-batch, not on every backward() call (8x reduction in sync ops)
            # -----------------------------------------------------------------------
            accumulate_context = accelerator.accumulate(student_model) if accelerator is not None else contextlib.nullcontext()
            with accumulate_context:
                # Progress logging for first few steps only
                should_log_detail = global_step < 3

                if should_log_detail:
                    print(f"\n[Step {global_step}, Micro-batch {micro_batch_count+1}/{accumulation_steps}] Batch received from dataloader...", flush=True)

                # ONLINE DISTILLATION: Move batch to teacher device first (GPU 0 has more free space)
                # compute_loss_with_kl will copy to student device when needed
                batch = {k: v.to(args.teacher_device) for k, v in batch.items()}

                if should_log_detail:
                    print(f"[Step {global_step}, Micro-batch {micro_batch_count+1}/{accumulation_steps}] Computing forward pass (teacher + student)...", flush=True)

                # Compute loss with KL (using dynamic temperature and kl_weight)
                # Teacher forward pass is computed inline (no precomputed outputs)
                loss, lm_loss, kl_loss, metrics = compute_loss_with_kl(
                    student_model, teacher_model, batch,
                    kl_weight=current_kl_weight,
                    temperature=current_temperature,
                    expected_layers=num_layers,  # Auto-detected from model config
                    validate_shapes=validate_shapes,
                    teacher_outputs=None  # Online mode: teacher forward pass inline
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

                # Backward pass (simple gradient accumulation, no DDP)
                loss.backward()

                # PERFORMANCE FIX: Accumulate losses as tensors (avoid CUDA sync on every micro-batch)
                # Only convert to float when logging (once per optimizer step, not 8x per step)
                accumulated_loss += loss.detach()  # loss is already scaled by accumulation_steps
                accumulated_lm_loss += lm_loss / accumulation_steps  # lm_loss is a tensor now
                accumulated_kl_loss += kl_loss / accumulation_steps  # kl_loss is a tensor now

                # Aggregate router metrics across micro-batches for this optimizer step
                for key in step_router_metrics:
                    step_router_metrics[key].extend(metrics[key])

                # Check if gradient accumulation is complete
                should_step = (micro_batch_count == accumulation_steps)
                if should_step:
                    if should_log_detail:
                        print(f"\n[Step {global_step + 1}] Gradient accumulation complete ({micro_batch_count} micro-batches). Running optimizer step...", flush=True)

                    # Gradient clipping
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
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
                        # No DDP reduction needed (single process)

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
                        # No DDP reduction needed (single process)
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

                            # No DDP broadcasting needed (single process)

                            # Add merge stats to metrics (if we saved metrics this step)
                            if step_metrics is not None:
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
                            model_to_save = student_model  # No DDP, no unwrapping needed

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
                        if accelerator is not None:
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

                    if global_step >= args.max_steps:
                        break

        # Detect dataset exhaustion (inner loop exited but we haven't reached max_steps)
        if global_step < args.max_steps:
            current_epoch += 1
            if should_print:
                print(f"\n{'='*80}")
                print(f"EPOCH {current_epoch} COMPLETE - Dataset exhausted, restarting from beginning")
                print(f"{'='*80}")
                print(f"  Steps completed: {global_step:,} / {args.max_steps:,}")
                print(f"  Batches in epoch: {batches_in_current_epoch:,}")
                if args.max_samples is not None:
                    print(f"  Samples per epoch: {args.max_samples:,}")
                    steps_per_epoch = args.max_samples // args.gradient_accumulation_steps
                    print(f"  Steps per epoch: ~{steps_per_epoch:,}")
                    remaining_epochs = (args.max_steps - global_step) / steps_per_epoch
                    print(f"  Estimated epochs remaining: {remaining_epochs:.1f}")
                print(f"  Starting epoch {current_epoch + 1}...")
                print(f"{'='*80}\n")
            batches_in_current_epoch = 0  # Reset for next epoch

    pbar.close()

    # Save final model (only on main process in DDP)
    if should_print:
        print("\nSaving final model...")
        final_dir = output_dir / "final"
        final_dir.mkdir(exist_ok=True)

        # Unwrap model from DDP if using Accelerate
        model_to_save = accelerator.unwrap_model(student_model) if accelerator is not None else student_model

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
    if accelerator is not None:
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
        description="NEXUS Online Distillation Training with SFA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ONLINE DISTILLATION (teacher GPU 0, student GPU 1, no DDP):
Teacher model loaded and runs forward pass during training. Simple gradient accumulation.

SFA (Sequential Fine-tuning with Averaging) mitigates catastrophic forgetting using
a ROLLING ANCHOR approach: merge with previous checkpoint (not initial checkpoint).

Example usage:
  python train.py \\
      --teacher-model /path/to/gpt-oss-120b \\
      --student-model /path/to/gpt-oss-120b-nexus \\
      --teacher-device cuda:0 \\
      --student-device cuda:1 \\
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
    parser.add_argument("--teacher-model", type=str, required=True,
                       help="Path to teacher model (base GPT-OSS model)")
    parser.add_argument("--student-model", type=str, required=True,
                       help="Path to student model with shared expert")

    # Training arguments
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0,
                       help="Gradient clipping value (default: 1.0). Set to large value (e.g., 1000) to effectively disable.")

    # Performance arguments
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory (enables longer sequences)")

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

    # Device arguments
    parser.add_argument("--teacher-device", type=str, default="cuda:0",
                       help="Device for teacher model (e.g., 'cuda:0'). For 120B models, "
                            "should typically be a different GPU than --student-device. "
                            "MXFP4 models require CUDA (CPU not supported).")
    parser.add_argument("--student-device", type=str, default="cuda:1",
                       help="Device for student model (e.g., 'cuda:1'). For 120B models, "
                            "should typically be a different GPU than --teacher-device. "
                            "MXFP4 models require CUDA (CPU not supported).")

    # Data arguments
    parser.add_argument("--dataset", type=str, default="nemotron",
                       choices=["nemotron", "c4"],
                       help="Dataset to use: 'nemotron' (code+math+tool) or 'c4' (English text)")
    parser.add_argument("--local-nemotron-code", type=str, default=None,
                       help="Path to local Nemotron code dataset (avoids download)")
    parser.add_argument("--local-nemotron-math", type=str, default=None,
                       help="Path to local Nemotron math dataset (avoids download)")
    parser.add_argument("--local-nemotron-tool", type=str, default=None,
                       help="Path to local Nemotron tool_calling dataset (avoids download)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of training samples per epoch (None = unlimited streaming, recommended)")
    parser.add_argument("--seq-len", type=int, default=512,
                       help="Maximum sequence length")

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

    # Run training
    train(args)


if __name__ == "__main__":
    main()