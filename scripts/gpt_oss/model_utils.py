#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
Model Utilities for GPT-OSS Training

Handles model loading, freezing, and configuration for training
routers + shared expert while keeping routed experts frozen.

Core functions:
- prepare_model_for_stage0: Freeze components for Stage-0 training
- set_deterministic_training: Set random seeds for reproducibility
- load_teacher_model: Load teacher with MXFP4 quantization
- load_student_model: Load student with MXFP4 quantization (DDP-aware)
- validate_device_config: Parse and validate device strings
"""

import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, set_seed


def set_deterministic_training(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace transformers seed

    # Set deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed to {seed} for reproducibility")


def prepare_model_for_stage0(model, freeze_router=False):
    """
    Freeze appropriate components for Stage-0 training.

    Args:
        model: Model to prepare
        freeze_router: If True, freeze router weights (train only shared expert)

    Trains (default): routers (~13M) + shared expert (~896M) = ~909M parameters
    Trains (freeze_router=True): shared expert only (~896M parameters)
    Freezes: MXFP4 routed experts, attention, embeddings, (optionally routers)
    """
    # Mark as PEFT to prevent issues
    model._hf_peft_config_loaded = True

    for name, param in model.named_parameters():
        # Freeze MXFP4 routed experts
        if "mlp.experts." in name:
            param.requires_grad = False
        # Optionally freeze routers (router.weight and router.bias)
        elif freeze_router and "mlp.router." in name:
            param.requires_grad = False
        # Freeze attention layers
        elif "self_attn" in name or "linear_attention" in name:
            param.requires_grad = False
        # Freeze embeddings and LM head
        elif "embed_tokens" in name or name == "lm_head.weight":
            param.requires_grad = False

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    if freeze_router:
        print(f"\n⚠️  Router FROZEN - Training shared expert only")
        print(f"   Simpler optimization, no router destabilization risk")

    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # List trainable components
    print("\nTrainable components:")
    trainable_components = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            component = name.split('.')[0:3]  # Get first 3 parts
            trainable_components.add('.'.join(component))

    for component in sorted(trainable_components):
        print(f"  - {component}")

    if freeze_router:
        print("\nFrozen components (in addition to experts/attention/embeddings):")
        print(f"  - model.layers.*.mlp.router")

    return model


def load_teacher_model(model_path, device_id):
    """
    Load teacher model with MXFP4 quantization.

    Args:
        model_path: Path to teacher model
        device_id: CUDA device ID (e.g., 0 for cuda:0)

    Returns:
        teacher_model: Loaded model in eval mode
    """
    print("\nLoading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device_id}  # MXFP4 requires loading directly to GPU
    )
    teacher_model.eval()
    print(f"  ✓ Teacher model loaded on cuda:{device_id}")
    return teacher_model


def load_student_model(model_path, device_id, use_ddp=False, local_rank=None):
    """
    Load student model with MXFP4 quantization (DDP-aware).

    Args:
        model_path: Path to student model
        device_id: CUDA device ID (e.g., 1 for cuda:1) - ignored if use_ddp=True
        use_ddp: Whether DDP is enabled
        local_rank: Local rank for DDP (required if use_ddp=True)

    Returns:
        student_model: Loaded model
    """
    print("\nLoading student model...")

    if use_ddp:
        # CRITICAL: In DDP mode, each process must load to its own GPU
        # Otherwise both processes try to load to GPU 0 simultaneously → OOM!
        if local_rank is None:
            raise ValueError("local_rank required for DDP mode")
        student_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": local_rank}  # Each process loads to its designated GPU
        )
        print(f"  ✓ Student model loaded on cuda:{local_rank} (DDP rank {local_rank})")
    else:
        # Single GPU mode: load to specified device
        student_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": device_id}  # MXFP4 requires loading directly to GPU
        )
        print(f"  ✓ Student model loaded on cuda:{device_id}")

    return student_model


def validate_device_config(teacher_device, student_device, precomputed=False):
    """
    Parse and validate device configuration.

    Args:
        teacher_device: Teacher device string (e.g., "cuda:0")
        student_device: Student device string (e.g., "cuda:1")
        precomputed: Whether using precomputed teacher outputs

    Returns:
        tuple: ((teacher_backend, teacher_device_id), (student_backend, student_device_id))

    Raises:
        ValueError: If device configuration is invalid
    """
    def parse_device(device_str):
        """Parse device string into (backend, device_id) tuple."""
        device_str = device_str.strip().lower()
        if device_str == "cpu":
            return "cpu", None
        if device_str.startswith("cuda"):
            if ":" in device_str:
                return "cuda", int(device_str.split(":")[1])
            return "cuda", 0
        raise ValueError(f"Unsupported device string: {device_str} (expected 'cpu' or 'cuda' or 'cuda:N')")

    # Parse devices
    teacher_backend, teacher_device_id = parse_device(teacher_device)
    student_backend, student_device_id = parse_device(student_device)

    # CRITICAL: MXFP4 quantization does not support CPU
    if not precomputed and teacher_backend == "cpu":
        raise ValueError(
            "Teacher cannot use CPU with MXFP4 quantization\n"
            f"  Current: --teacher-device {teacher_device}\n"
            "  Fix: Use --teacher-device cuda:0 (or another GPU)"
        )

    if student_backend == "cpu":
        raise ValueError(
            "Student cannot use CPU with MXFP4 quantization\n"
            f"  Current: --student-device {student_device}\n"
            "  Fix: Use --student-device cuda:1 (or another GPU)"
        )

    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs detected. MXFP4 models require GPU.")

    if teacher_device_id is not None and teacher_device_id >= num_gpus:
        raise ValueError(f"Teacher device cuda:{teacher_device_id} not available (only {num_gpus} GPU(s) detected)")
    if student_device_id >= num_gpus:
        raise ValueError(f"Student device cuda:{student_device_id} not available (only {num_gpus} GPU(s) detected)")

    # Warn if both models on same GPU (high OOM risk)
    if not precomputed and teacher_device_id == student_device_id:
        print("\n" + "="*80)
        print(f"WARNING: Teacher and student are both on cuda:{teacher_device_id}")
        print("="*80)
        print("  HIGH RISK OF OUT-OF-MEMORY (OOM)!")
        print("  Each 120B model needs ~75-80GB VRAM")
        print("  RECOMMENDATION: Use separate GPUs")
        print("    --teacher-device cuda:0 --student-device cuda:1")
        print("  OR: Use offline distillation (precomputed teacher outputs)")
        print("="*80 + "\n")

    # Print device configuration
    if not precomputed:
        print("\nDevice Configuration (Online Distillation):")
        print(f"  Teacher will use: cuda:{teacher_device_id}")
    else:
        print("\nDevice Configuration (Offline Distillation):")
        print(f"  Teacher: Precomputed outputs (no model in memory)")
    print(f"  Student will use: cuda:{student_device_id}")

    return (teacher_backend, teacher_device_id), (student_backend, student_device_id)


def setup_gradient_checkpointing(model, enabled=True):
    """
    Enable/disable gradient checkpointing for memory savings.

    Args:
        model: Model to configure
        enabled: Whether to enable gradient checkpointing

    Returns:
        model: Configured model
    """
    if enabled:
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled (saves memory, slight compute overhead)")
    else:
        model.gradient_checkpointing_disable()
        print("✓ Gradient checkpointing disabled (faster, more memory)")
    return model


__all__ = [
    'set_deterministic_training',
    'prepare_model_for_stage0',
    'load_teacher_model',
    'load_student_model',
    'validate_device_config',
    'setup_gradient_checkpointing'
]
