#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
Knowledge Distillation Utilities

Implements temperature-scaled KL divergence for router distillation
from teacher to student models. Supports both online and offline modes.

Core functions:
- validate_router_logits: Validate router logits shape and count
- compute_kl_with_temperature: Temperature-scaled KL divergence (Hinton et al., 2015)
- compute_router_metrics: L1 distance, entropy monitoring
- compute_loss_with_kl: Main loss computation (supports online/offline distillation)
"""

import warnings
import torch
import torch.nn.functional as F


def validate_router_logits(router_logits, expected_layers, model_name="model"):
    """
    Validate router logits shape and count.

    Args:
        router_logits: List of router logits tensors
        expected_layers: Expected number of layers (24 for 20B, 36 for 120B)
        model_name: Name for error messages

    Returns:
        bool: True if valid, raises warnings/errors otherwise
    """
    if router_logits is None:
        warnings.warn(
            f"{model_name}: router_logits is None. "
            "Model may not be configured with output_router_logits=True"
        )
        return False

    if not isinstance(router_logits, (list, tuple)):
        warnings.warn(
            f"{model_name}: router_logits is not a list/tuple, got {type(router_logits)}"
        )
        return False

    num_layers = len(router_logits)
    if num_layers == 0:
        warnings.warn(f"{model_name}: router_logits list is empty")
        return False

    if num_layers != expected_layers:
        warnings.warn(
            f"{model_name}: Expected {expected_layers} layers of router logits, "
            f"but got {num_layers}"
        )

    # Check first layer shape
    first_logits = router_logits[0]
    if not isinstance(first_logits, torch.Tensor):
        warnings.warn(
            f"{model_name}: router_logits[0] is not a tensor, got {type(first_logits)}"
        )
        return False

    # Expected shape: (batch_size, seq_len, num_experts) or (batch_size*seq_len, num_experts)
    if len(first_logits.shape) == 2:
        # Flattened format: (batch*seq, num_experts) - common in some models
        tokens, num_experts = first_logits.shape
        # Cannot validate batch/seq split without external info, but shape is valid
        # Just check consistency across layers
        for i, logits in enumerate(router_logits):
            if logits.shape != (tokens, num_experts):
                warnings.warn(
                    f"{model_name}: Layer {i} router logits shape {logits.shape} "
                    f"doesn't match expected {(tokens, num_experts)}"
                )
                return False
        return True

    elif len(first_logits.shape) == 3:
        # Standard 3D format: (batch, seq, num_experts)
        batch_size, seq_len, num_experts = first_logits.shape

        # Validate all layers have consistent shape
        for i, logits in enumerate(router_logits):
            if logits.shape != (batch_size, seq_len, num_experts):
                warnings.warn(
                    f"{model_name}: Layer {i} router logits shape {logits.shape} "
                    f"doesn't match expected {(batch_size, seq_len, num_experts)}"
                )
                return False
        return True

    else:
        warnings.warn(
            f"{model_name}: Expected 2D (tokens, experts) or 3D (batch, seq, experts) router logits, "
            f"got shape {first_logits.shape}"
        )
        return False


def compute_kl_with_temperature(student_logits, teacher_logits, temperature=4.0):
    """
    Compute KL divergence with temperature scaling for knowledge distillation.

    Temperature T > 1 softens the probability distributions, making them more informative
    for learning (Hinton et al., 2015: "Distilling the Knowledge in a Neural Network").

    Args:
        student_logits: Student router logits (batch, seq_len, num_experts)
        teacher_logits: Teacher router logits (batch, seq_len, num_experts)
        temperature: Temperature for softening distributions (default: 4.0)
                    Higher T = softer distributions = more emphasis on dark knowledge
                    Lower T = sharper distributions = more emphasis on hard predictions

    Returns:
        KL divergence scaled by T^2

    Note on T^2 scaling:
        The T^2 factor maintains gradient magnitude across different temperatures.
        Without it, gradients would vanish as T increases. This is standard practice
        in knowledge distillation literature (Hinton et al., 2015).

        Mathematically: KL(P||Q) with temperature T should contribute equally to the
        gradient as KL without temperature, hence the T^2 compensation.
    """
    # Scale logits by temperature to soften distributions
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # Compute KL and scale by T^2 to maintain gradient magnitude
    kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    kl = kl * (temperature ** 2)

    # Clamp to ensure KL is non-negative (mathematical property: KL(P||Q) ≥ 0)
    # Small negative values (~1e-7) can occur due to floating point precision when
    # distributions are identical. Clamping is safe because:
    # 1. KL divergence is always non-negative by definition (information theory)
    # 2. Negative values are numerical artifacts, not true divergence
    # 3. This prevents rare NaN issues in downstream gradient computations
    kl = torch.clamp(kl, min=0.0)

    return kl


def compute_router_metrics(student_logits, teacher_logits):
    """
    Compute metrics to monitor router divergence.

    PERFORMANCE FIX: Returns detached tensors (not Python floats) to avoid CUDA sync.
    Only call .item() when logging, not on every micro-batch (864 syncs/step → 3 syncs/log).

    Returns:
        - l1_distance: L1 distance between routing distributions (detached tensor)
        - student_entropy: Entropy of student routing distribution (detached tensor)
        - teacher_entropy: Entropy of teacher routing distribution (detached tensor)
    """
    # Metrics should not create gradients or retain the computation graph.
    with torch.no_grad():
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)

        # L1 distance between distributions (keep as tensor, detach to free graph)
        l1_distance = torch.abs(student_probs - teacher_probs).sum(dim=-1).mean().detach()

        # Entropy of distributions (higher = more uniform, lower = more peaked)
        # Clamp probabilities to avoid log(0) and improve numerical stability.
        eps = 1e-12
        student_entropy = -(student_probs * torch.log(student_probs.clamp_min(eps))).sum(dim=-1).mean().detach()
        teacher_entropy = -(teacher_probs * torch.log(teacher_probs.clamp_min(eps))).sum(dim=-1).mean().detach()

    return l1_distance, student_entropy, teacher_entropy


def compute_loss_with_kl(student_model, teacher_model, batch, kl_weight=1.0, temperature=4.0,
                         expected_layers=36, validate_shapes=True, teacher_outputs=None):
    """
    Compute loss with temperature-scaled KL distillation.

    Supports both online and offline distillation:
    - Online: teacher_model performs forward pass
    - Offline: teacher_outputs provided (precomputed)

    Args:
        student_model: Student model with shared expert
        teacher_model: Teacher model (original) - can be None if teacher_outputs provided
        batch: Input batch
        kl_weight: Weight for KL loss (default: 1.0 for equal importance with LM loss)
        temperature: Temperature for KL scaling (default: 4.0)
        expected_layers: Expected number of layers (24 for 20B, 36 for 120B)
        validate_shapes: Whether to validate router logits shapes (default: True)
        teacher_outputs: Precomputed teacher outputs (optional, for offline distillation)

    Returns:
        total_loss: Combined loss for backpropagation
        lm_loss: Language modeling loss
        kl_loss: Average KL divergence across layers
        metrics: Dictionary of monitoring metrics
    """
    # Student forward pass with router logits
    # VRAM OPTIMIZATION: Batch may be on teacher device (GPU 0), copy to student device if needed
    student_device = next(student_model.parameters()).device
    batch_on_student = {k: v.to(student_device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    student_outputs = student_model(**batch_on_student, output_router_logits=True)
    lm_loss = student_outputs.loss

    # Initialize metrics
    metrics = {
        'layer_kl': [],
        'layer_l1': [],
        'student_entropy': [],
        'teacher_entropy': []
    }

    # Short-circuit if we are not using KL distillation
    kl_loss = 0.0
    if kl_weight > 0.0:
        # Get teacher outputs (either from model or precomputed)
        if teacher_outputs is None:
            # Online distillation: Teacher forward pass (no gradients needed)
            with torch.no_grad():
                # VRAM OPTIMIZATION: Batch is already on teacher device, no need to copy
                # Teacher does not need labels; avoid extra LM loss computation
                teacher_batch = {k: v for k, v in batch.items() if k != "labels"}
                teacher_outputs = teacher_model(**teacher_batch, output_router_logits=True)
        # Else: teacher_outputs already provided (offline distillation)

        # Validate router logits if requested
        student_valid = teacher_valid = True
        if validate_shapes:
            student_valid = validate_router_logits(
                student_outputs.router_logits, expected_layers, "Student"
            )
            teacher_valid = validate_router_logits(
                teacher_outputs.router_logits, expected_layers, "Teacher"
            )

            if not student_valid or not teacher_valid:
                warnings.warn(
                    "Router logits validation failed. KL loss will be zero. "
                    "Check that models are configured correctly."
                )

        # Compute KL divergence with temperature scaling
        if (
            student_valid
            and teacher_valid
            and student_outputs.router_logits
            and teacher_outputs.router_logits
        ):
            num_layers = min(
                len(student_outputs.router_logits),
                len(teacher_outputs.router_logits),
            )

            # VRAM OPTIMIZATION: Do KL computation on teacher's device (GPU 0) instead of student's (GPU 1)
            # This avoids copying 36 layers of teacher router logits to the constrained GPU
            teacher_device = teacher_outputs.router_logits[0].device

            for layer_idx in range(num_layers):
                student_logits = student_outputs.router_logits[layer_idx]
                teacher_logits = teacher_outputs.router_logits[layer_idx]

                # Move student logits to teacher device for KL computation (saves GPU 1 memory)
                student_logits_on_teacher = student_logits.to(teacher_device)

                # Temperature-scaled KL for this layer (computed on GPU 0)
                layer_kl = compute_kl_with_temperature(
                    student_logits_on_teacher, teacher_logits, temperature
                )
                kl_loss += layer_kl

                # Compute monitoring metrics for this layer (on GPU 0)
                l1_dist, s_entropy, t_entropy = compute_router_metrics(
                    student_logits_on_teacher, teacher_logits
                )

                # PERFORMANCE FIX: Store tensors, not Python floats (avoid CUDA sync)
                # Move to CPU to free GPU memory (metrics are only for logging, not compute)
                metrics['layer_kl'].append(layer_kl.detach().cpu())
                metrics['layer_l1'].append(l1_dist.cpu())
                metrics['student_entropy'].append(s_entropy.cpu())
                metrics['teacher_entropy'].append(t_entropy.cpu())

            kl_loss = kl_loss / num_layers

            # Move KL loss back to student device for combining with LM loss
            kl_loss = kl_loss.to(lm_loss.device)

    # Combine losses
    total_loss = lm_loss + kl_weight * kl_loss

    # PERFORMANCE FIX: Return losses as detached tensors (convert to float only when logging)
    return total_loss, lm_loss.detach(), kl_loss.detach() if isinstance(kl_loss, torch.Tensor) else kl_loss, metrics


__all__ = [
    'validate_router_logits',
    'compute_kl_with_temperature',
    'compute_router_metrics',
    'compute_loss_with_kl'
]
