#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
SFA (Sequential Fine-tuning with Averaging) for Continual Learning

Implements rolling anchor approach to mitigate catastrophic forgetting:
- Periodically merge current weights with previous checkpoint
- Memory-efficient: loads only trainable params (~900MB vs 60GB)
- DDP-safe: rank 0 merges, broadcasts to all ranks

Based on: "Soup to go: mitigating forgetting during continual learning
with model averaging" (arXiv:2501.05559)

Core functions:
- load_trainable_weights_from_checkpoint: Fast path for loading checkpoint
- merge_with_anchor: In-place merge with previous checkpoint
- save_trainable_weights_safetensors: Save lightweight file for fast SFA
- store_anchor_checkpoint: Reference implementation (not used in rolling anchor)
"""

import torch
from pathlib import Path


def store_anchor_checkpoint(model):
    """
    Store a deep copy of the model's state dict as the anchor for SFA merging.

    ⚠️  NOTE: This function is NOT used by the current rolling anchor implementation.
    It is kept for reference to show the alternative fixed-anchor approach.

    CURRENT IMPLEMENTATION (Rolling Anchor):
        The rolling anchor approach loads previous checkpoints from disk at each merge
        point instead of storing an initial anchor in memory. See the training loop for
        the actual implementation used in training.

    ALTERNATIVE IMPLEMENTATION (Fixed Anchor - this function):
        For fixed-anchor SFA (always merge with step 0):
            anchor_state, trainable_param_names = store_anchor_checkpoint(student_model)
            # Then merge with anchor_state at each merge point

        Pros: Simpler logic, maintains connection to initial checkpoint
        Cons: Too conservative, prevents gradual drift, less effective for continual learning

    The rolling anchor approach is preferred because it allows gradual adaptation while
    still preventing catastrophic forgetting.

    Returns:
        anchor_state_dict: Deep copy of model's state dict (on CPU to save GPU memory)
        trainable_param_names: Set of parameter names that are trainable
    """
    print("\nStoring anchor checkpoint for SFA...")
    anchor_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Track which parameters are trainable for accurate merge statistics
    trainable_param_names = {name for name, param in model.named_parameters() if param.requires_grad}

    print(f"✓ Anchor checkpoint stored ({len(anchor_state)} parameters)")
    print(f"  Trainable parameters tracked: {len(trainable_param_names)}")
    return anchor_state, trainable_param_names


def load_trainable_weights_from_checkpoint(checkpoint_dir, trainable_param_names):
    """
    Load ONLY trainable weights from checkpoint (memory-efficient).

    For GPT-OSS-120B Stage-0: 60GB full model → ~900MB trainable weights (65x reduction)
    This is much faster and more memory-efficient than loading the full model.

    Strategy:
    1. Try loading from trainable_weights.safetensors (FAST: loads pre-extracted ~900MB file)
    2. Fallback to loading from full model shards (SLOW: filters trainable params from 60GB)

    The fallback ensures compatibility with checkpoints created before we added the
    trainable_weights.safetensors optimization.

    Args:
        checkpoint_dir: Path to checkpoint directory
        trainable_param_names: Set of parameter names that are trainable

    Returns:
        dict: State dict containing only trainable parameters (on CPU)

    Raises:
        FileNotFoundError: If checkpoint directory or required files don't exist
    """
    from safetensors.torch import load_file
    import json

    checkpoint_path = Path(checkpoint_dir)

    # Try fast path: load from trainable_weights.safetensors
    trainable_weights_file = checkpoint_path / "trainable_weights.safetensors"

    if trainable_weights_file.exists():
        print(f"  Loading trainable weights from trainable_weights.safetensors (FAST PATH)...")
        trainable_state_dict = load_file(trainable_weights_file, device='cpu')
        print(f"  ✓ Loaded {len(trainable_state_dict)} trainable parameters (~{sum(p.numel() for p in trainable_state_dict.values()) / 1e6:.1f}M params)")
        return trainable_state_dict

    # Fallback: load from full model shards (slower but works for older checkpoints)
    print(f"  trainable_weights.safetensors not found, loading from full model shards (SLOW PATH)...")

    index_file = checkpoint_path / "model.safetensors.index.json"

    if not index_file.exists():
        raise FileNotFoundError(f"Checkpoint index not found: {index_file}")

    # Load index to find which shards contain which parameters
    with open(index_file, 'r') as f:
        index = json.load(f)

    weight_map = index['weight_map']

    # Find which shards we need to load (only those with trainable params)
    shards_to_load = set()
    trainable_keys_to_load = {}  # key -> shard mapping

    for param_name in trainable_param_names:
        if param_name in weight_map:
            shard_file = weight_map[param_name]
            shards_to_load.add(shard_file)
            trainable_keys_to_load[param_name] = shard_file

    print(f"  Loading {len(trainable_keys_to_load)} trainable parameters from {len(shards_to_load)} shards...")

    # Load only trainable parameters from each shard
    trainable_state_dict = {}

    for shard_file in sorted(shards_to_load):
        shard_path = checkpoint_path / shard_file
        # Load entire shard (we need to, safetensors doesn't support partial loading per file)
        shard_data = load_file(shard_path, device='cpu')

        # Extract only trainable params from this shard
        for key in shard_data.keys():
            if key in trainable_param_names:
                trainable_state_dict[key] = shard_data[key].clone()

        # Delete shard to free memory
        del shard_data

    print(f"  ✓ Loaded {len(trainable_state_dict)} trainable parameters (~{sum(p.numel() for p in trainable_state_dict.values()) / 1e6:.1f}M params)")

    return trainable_state_dict


def merge_with_anchor(model, anchor_state_dict, alpha=0.25):
    """
    Merge current model weights with anchor checkpoint in-place.

    MEMORY OPTIMIZED: Direct parameter updates avoid creating full state_dict copies.
    Old implementation wasted ~60GB copying frozen params; new version touches only trainable params.

    Implements: merged = alpha * anchor + (1 - alpha) * current

    Args:
        model: Current model (will be modified in-place, on GPU)
        anchor_state_dict: Dict containing ONLY trainable parameters from anchor (on CPU).
                          The invariant that this contains only trainable params is enforced
                          by the caller (load_trainable_weights_from_checkpoint).
        alpha: Weight for anchor (0 = keep current, 1 = revert to anchor)
               ⚠️  IMPORTANT: Higher alpha = MORE conservative (retains MORE of anchor)

               Alpha Guidelines:
               - 0.1-0.2: Mostly keep new learning, light regularization (aggressive learning)
               - 0.25: Balanced (recommended default for most cases)
               - 0.3-0.4: Conservative, stronger retention of anchor (cautious learning)
               - 0.5: Equal weighting (maximum regularization)

               Example: alpha=0.25 means 25% anchor + 75% current = mild pull toward anchor

    Returns:
        merge_stats: Statistics about the merge
        merged_state_dict: Merged weights (on CPU) for DDP broadcasting
            - 'trainable_params_merged': Number of parameters merged
            - 'max_change': Maximum change in any parameter
            - 'mean_change': Average change across all parameters
    """
    # Track merge statistics
    merge_stats = {
        'trainable_params_merged': 0,
        'max_change': 0.0,
        'mean_change': 0.0,
    }

    param_change_means = []  # Track mean change per parameter (for aggregate stats)
    merged_state_dict = {}  # Build merged state for broadcasting (avoids state_dict() call later)

    print(f"  Merging {len(anchor_state_dict)} trainable parameters (in-place CPU→GPU copy)...")

    # MEMORY OPTIMIZATION: Iterate through parameters directly, not via state_dict()
    # This avoids creating a ~60GB copy of the entire model including frozen params
    for name, param in model.named_parameters():
        # Skip non-trainable parameters (frozen experts, attention, embeddings)
        if not param.requires_grad:
            continue

        # Check if this parameter exists in the anchor
        if name not in anchor_state_dict:
            print(f"  Warning: Trainable param '{name}' not in anchor checkpoint")
            continue

        # 1. Get current parameter data (move to CPU for merge calculation)
        #    Use .data to avoid gradient tracking during merge operation
        current_param_cpu = param.data.cpu()
        anchor_param_cpu = anchor_state_dict[name]

        # 2. Compute merge on CPU (saves GPU memory and compute)
        #    merged = alpha * anchor + (1 - alpha) * current
        merged_param_cpu = alpha * anchor_param_cpu + (1 - alpha) * current_param_cpu

        # 3. Track statistics (mean absolute change for this parameter)
        with torch.no_grad():
            diff = (merged_param_cpu - current_param_cpu).abs()
            param_mean_change = diff.mean().item()
            param_change_means.append(param_mean_change)
            merge_stats['trainable_params_merged'] += merged_param_cpu.numel()

        # 4. Store merged weights for broadcasting (on CPU, already computed)
        #    This avoids calling state_dict() later (which would waste 60GB)
        merged_state_dict[name] = merged_param_cpu.clone()

        # 5. IN-PLACE UPDATE: Copy merged values directly to GPU parameter
        #    copy_() handles CPU→GPU transfer and preserves parameter identity
        #    This is crucial for DDP: updates .data without breaking gradient hooks
        param.data.copy_(merged_param_cpu)

    # Compute aggregate statistics across all parameters
    if param_change_means:
        merge_stats['max_change'] = max(param_change_means)
        merge_stats['mean_change'] = sum(param_change_means) / len(param_change_means)

    return merge_stats, merged_state_dict


def save_trainable_weights_safetensors(model, output_dir):
    """
    Save only trainable weights as separate safetensors file for fast SFA merging.

    This creates a lightweight file (~900MB for GPT-OSS-120B) containing only the
    trainable parameters, which is much faster to load for SFA merging than the
    full 60GB checkpoint.

    Args:
        model: PyTorch model with some parameters marked as trainable
        output_dir: Directory to save trainable_weights.safetensors

    Returns:
        tuple: (num_params_saved, total_param_count)
    """
    from safetensors.torch import save_file

    trainable_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.cpu().clone()

    output_path = Path(output_dir)
    trainable_weights_path = output_path / "trainable_weights.safetensors"
    save_file(trainable_state, trainable_weights_path)

    num_params = sum(p.numel() for p in trainable_state.values())
    print(f"  ✓ Saved {len(trainable_state)} trainable parameters (~{num_params / 1e6:.1f}M params) to trainable_weights.safetensors")

    return len(trainable_state), num_params


__all__ = [
    'store_anchor_checkpoint',
    'load_trainable_weights_from_checkpoint',
    'merge_with_anchor',
    'save_trainable_weights_safetensors'
]
