#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
GPT-OSS Training Metrics Visualization

Plot Stage-0 training metrics from log output.

Enhanced to support advanced scheduler metrics:
- Temperature annealing
- KL weight warmup
- Router bias scheduling
- Multi-phase LR schedule (warmup → stable → decay)

Usage:
    # From log file (auto-extracts Step lines):
    python plot_training_metrics.py --log outputs/stage0_production_part2.log --output metrics.png

    # From pasted data:
    python plot_training_metrics.py --data data.txt --output metrics.png

    # Interactive (paste data when prompted):
    python plot_training_metrics.py --output metrics.png

    # Load from metrics.jsonl (comprehensive):
    python plot_training_metrics.py --jsonl outputs/stage0/metrics.jsonl --output metrics.png
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import re
import sys
import json
from pathlib import Path


def parse_step_line(line):
    """Parse a single step line into metrics dict."""
    # Example (new): "Step 10, epoch=0: loss=5.8355, lm=5.6728, kl=0.1627, ..."
    # Example (old): "Step 10: loss=5.8355, lm=5.6728, kl=0.1627, ..."

    # Extract step number (flexible pattern supports both formats)
    step_match = re.search(r'Step (\d+)', line)
    if not step_match:
        return None
    step = int(step_match.group(1))

    # Extract epoch (optional, for new format)
    epoch_match = re.search(r'epoch=(\d+)', line)
    epoch = int(epoch_match.group(1)) if epoch_match else 0

    # Extract metrics
    metrics = {}
    patterns = {
        'loss': r'loss=([\d.]+)',
        'lm': r'lm=([\d.]+)',
        'kl': r'kl=([\d.]+)',
        'l1_dist': r'l1_dist=([\d.]+)',
        's_entropy': r's_entropy=([\d.]+)',
        't_entropy': r't_entropy=([\d.]+)',
        'grad_norm': r'grad_norm=([\d.]+)',
        'lr': r'lr=([\d.e\-+]+)',
        'temperature': r'(?:temperature|temp)=([\d.]+)',  # Support both temperature= and temp=
        'kl_weight': r'kl_weight=([\d.]+)',      # New: advanced scheduler
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            metrics[key] = float(match.group(1))
        elif key in ['loss', 'lm', 'kl', 'lr']:
            # Core metrics must be present
            return None
        # Optional metrics (grad_norm, temperature, kl_weight) can be missing

    # Add epoch to metrics for tracking dataset passes
    metrics['epoch'] = epoch

    return step, metrics


def load_data_from_jsonl(jsonl_path):
    """Load metrics from metrics.jsonl file."""
    steps = []
    all_metrics = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if 'step' not in data:
                    continue

                steps.append(data['step'])
                all_metrics.append(data)
            except json.JSONDecodeError:
                continue

    return steps, all_metrics


def parse_gradient_clip_line(line):
    """
    Parse gradient clipping event line.

    Formats supported:
    - New: GRADIENT_CLIP: step={step}, norm={norm}, threshold={threshold}, ratio={ratio}x
    - Old: Gradient norm: {norm} (clipped to {threshold}, ratio: {ratio}x)

    Returns:
        (step, norm, threshold) tuple or None (step=None for old format)
    """
    # New format with explicit step
    if 'GRADIENT_CLIP:' in line:
        try:
            step_match = re.search(r'step=(\d+)', line)
            norm_match = re.search(r'norm=([\d.]+)', line)
            threshold_match = re.search(r'threshold=([\d.]+)', line)

            if step_match and norm_match and threshold_match:
                return (
                    int(step_match.group(1)),
                    float(norm_match.group(1)),
                    float(threshold_match.group(1))
                )
        except:
            pass

    # Old format: "Gradient norm: X.XXXX (clipped to Y.Y, ratio: Z.ZZx)"
    if 'Gradient norm:' in line and 'clipped to' in line:
        try:
            norm_match = re.search(r'Gradient norm:\s*([\d.]+)', line)
            threshold_match = re.search(r'clipped to\s*([\d.]+)', line)

            if norm_match and threshold_match:
                # Try to extract step from same line or nearby context
                step_match = re.search(r'(\d+)/\d+\s*\[', line)  # From tqdm progress bar
                step = int(step_match.group(1)) if step_match else None

                return (
                    step,
                    float(norm_match.group(1)),
                    float(threshold_match.group(1))
                )
        except:
            pass

    return None


def load_data_from_log(log_path):
    """Load metrics, gradient clipping events, and training events from log file."""
    steps = []
    all_metrics = []
    seen_steps = set()

    clip_events = []  # List of (step, pre_clip_norm, threshold) tuples
    sfa_merge_steps = []  # Steps where SFA merge occurred
    checkpoint_steps = []  # Steps where checkpoint was saved

    with open(log_path, 'r') as f:
        for line in f:
            # Parse SFA merge events
            if 'SFA MERGE at Step' in line:
                step_match = re.search(r'Step (\d+)', line)
                if step_match:
                    step = int(step_match.group(1))
                    sfa_merge_steps.append(step)
                    continue

            # Parse checkpoint save events
            if 'Saving checkpoint at step' in line:
                step_match = re.search(r'step (\d+)', line)
                if step_match:
                    step = int(step_match.group(1))
                    checkpoint_steps.append(step)
                    continue

            # Parse gradient clipping events
            clip_result = parse_gradient_clip_line(line)
            if clip_result is not None:
                clip_events.append(clip_result)
                continue

            # Parse regular step metrics
            if 'Step' not in line or 'loss=' not in line:
                continue

            result = parse_step_line(line)
            if result is None:
                continue

            step, metrics = result

            # Skip duplicate lines (each step is logged twice in current script)
            if step in seen_steps:
                continue

            seen_steps.add(step)
            steps.append(step)
            all_metrics.append(metrics)

    return steps, all_metrics, clip_events, sfa_merge_steps, checkpoint_steps


def load_data_from_text(text):
    """Load metrics from pasted text."""
    steps = []
    all_metrics = []
    seen_steps = set()

    for line in text.strip().split('\n'):
        if not line.strip():
            continue

        result = parse_step_line(line)
        if result is None:
            continue

        step, metrics = result

        # Skip duplicates
        if step in seen_steps:
            continue

        seen_steps.add(step)
        steps.append(step)
        all_metrics.append(metrics)

    return steps, all_metrics


def detect_schedule_phases(steps, lrs):
    """
    Detect warmup, stable, and decay phases from LR schedule.

    Uses improved heuristics:
    - Warmup end: LR reaches within 95% of peak and stays there
    - Stable end: LR drops below 95% of peak (cosine decay starts)

    Returns:
        dict with 'warmup_end', 'stable_end' step numbers (or None)
    """
    if len(lrs) < 10:
        return {'warmup_end': None, 'stable_end': None}

    phases = {'warmup_end': None, 'stable_end': None}

    peak_lr = max(lrs)
    peak_threshold = peak_lr * 0.95  # Within 95% of peak

    # Detect warmup end: First time LR reaches 95% of peak AND stays there
    for i, lr in enumerate(lrs):
        if lr >= peak_threshold and i > 10:  # Skip first 10 steps
            # Verify it stays at peak for at least 5 more steps (stable phase)
            if i + 5 < len(lrs):
                next_5_avg = sum(lrs[i:i+5]) / 5
                if next_5_avg >= peak_threshold * 0.98:  # Still at peak
                    phases['warmup_end'] = steps[i]
                    break
            elif i > len(lrs) * 0.8:  # Near end of training, accept it
                phases['warmup_end'] = steps[i]
                break

    # Detect stable end: LR drops below 95% of peak (decay starts)
    if phases['warmup_end']:
        warmup_idx = steps.index(phases['warmup_end'])
        # Look for sustained drop below peak threshold
        for i in range(warmup_idx + 10, len(lrs)):  # Skip a few steps after warmup
            if lrs[i] < peak_threshold:
                # Verify it stays low (not just noise)
                if i + 5 < len(lrs) and all(lrs[j] < peak_threshold for j in range(i, min(i+5, len(lrs)))):
                    phases['stable_end'] = steps[i]
                    break

    # Sanity check: If warmup_end is too early (< 10% of total steps), likely false detection
    # In that case, clear it to avoid misleading annotations
    if phases['warmup_end'] and len(steps) > 100:
        warmup_end_pct = phases['warmup_end'] / max(steps)
        if warmup_end_pct < 0.05:  # Warmup "ended" in first 5% - probably wrong
            phases['warmup_end'] = None
            phases['stable_end'] = None

    return phases


def add_event_markers(ax, steps, sfa_merge_steps=None, checkpoint_steps=None):
    """
    Add vertical lines to mark training events.

    Args:
        ax: Matplotlib axis to add markers to
        steps: All training steps (for x-axis range)
        sfa_merge_steps: List of steps where SFA merge occurred
        checkpoint_steps: List of steps where checkpoint was saved (excluding SFA merges)
    """
    if not sfa_merge_steps and not checkpoint_steps:
        return

    # Get y-axis limits for full-height lines
    y_min, y_max = ax.get_ylim()

    # SFA merge markers (red dashed lines)
    if sfa_merge_steps:
        for step in sfa_merge_steps:
            if min(steps) <= step <= max(steps):
                ax.axvline(x=step, color='red', linestyle='--', linewidth=1.5,
                          alpha=0.6, zorder=1)

    # Checkpoint-only markers (blue dotted lines, exclude SFA merge steps)
    if checkpoint_steps:
        sfa_set = set(sfa_merge_steps) if sfa_merge_steps else set()
        checkpoint_only = [s for s in checkpoint_steps if s not in sfa_set]
        for step in checkpoint_only:
            if min(steps) <= step <= max(steps):
                ax.axvline(x=step, color='blue', linestyle=':', linewidth=1,
                          alpha=0.4, zorder=1)


def plot_metrics(steps, all_metrics, output_path, title_suffix="",
                clip_events=None, sfa_merge_steps=None, checkpoint_steps=None):
    """Create comprehensive training metrics plot with event markers."""

    # Extract metric lists
    losses = [m.get('loss', 0) for m in all_metrics]
    lm_losses = [m.get('lm_loss', m.get('lm', 0)) for m in all_metrics]
    kl_losses = [m.get('kl_loss', m.get('kl', 0)) for m in all_metrics]
    l1_dists = [m.get('avg_l1_distance', m.get('l1_dist', 0)) for m in all_metrics]
    s_entropies = [m.get('avg_student_entropy', m.get('s_entropy', 0)) for m in all_metrics]
    t_entropies = [m.get('avg_teacher_entropy', m.get('t_entropy', 0)) for m in all_metrics]
    lrs = [m.get('learning_rate', m.get('lr', 0)) for m in all_metrics]

    # Extract optional metrics
    grad_norms = []
    temperatures = []
    kl_weights = []

    for m in all_metrics:
        grad_norms.append(m.get('gradient_norm', m.get('grad_norm')))
        temperatures.append(m.get('temperature'))
        kl_weights.append(m.get('kl_weight'))

    # Check if optional metrics are available
    has_grad_norm = any(g is not None for g in grad_norms)
    has_temperature = any(t is not None for t in temperatures)
    has_kl_weight = any(w is not None for w in kl_weights)
    has_advanced = has_temperature or has_kl_weight

    # Detect schedule phases
    phases = detect_schedule_phases(steps, lrs)

    # Determine grid layout
    if has_advanced:
        # 4x3 layout for advanced scheduler
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    elif has_grad_norm:
        # 3x3 layout for standard + grad norm
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    else:
        # 2x3 layout for basic metrics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    fig.suptitle(f'Stage-0 Training Metrics{title_suffix}', fontsize=16, fontweight='bold')

    # Plot 1: Total Loss
    axes[0, 0].plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=3, label='Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss (LM + KL)')
    axes[0, 0].grid(True, alpha=0.3)

    # Add event markers with legend
    add_event_markers(axes[0, 0], steps, sfa_merge_steps, checkpoint_steps)
    if sfa_merge_steps or checkpoint_steps:
        # Add legend entries for event markers (only on first plot)
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='b', linewidth=2, label='Loss')]
        if sfa_merge_steps:
            legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
                                        alpha=0.6, label=f'SFA Merge (n={len(sfa_merge_steps)})'))
        if checkpoint_steps:
            # Count checkpoint-only saves (excluding SFA merges)
            sfa_set = set(sfa_merge_steps) if sfa_merge_steps else set()
            ckpt_only_count = len([s for s in checkpoint_steps if s not in sfa_set])
            if ckpt_only_count > 0:
                legend_elements.append(Line2D([0], [0], color='blue', linestyle=':', linewidth=1,
                                            alpha=0.4, label=f'Checkpoint (n={ckpt_only_count})'))
        axes[0, 0].legend(handles=legend_elements, loc='best')

    # Plot 2: LM Loss
    axes[0, 1].plot(steps, lm_losses, 'g-', linewidth=2, marker='o', markersize=3)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('LM Loss')
    axes[0, 1].set_title('Language Modeling Loss')
    axes[0, 1].grid(True, alpha=0.3)
    add_event_markers(axes[0, 1], steps, sfa_merge_steps, checkpoint_steps)

    # Plot 3: KL Loss
    axes[0, 2].plot(steps, kl_losses, 'r-', linewidth=2, marker='o', markersize=3)
    axes[0, 2].axhline(y=0.09, color='orange', linestyle='--', label='Target ~0.09', alpha=0.7)
    axes[0, 2].axhline(y=0.05, color='red', linestyle='--', label='Collapse threshold', alpha=0.5)
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('KL Divergence')
    axes[0, 2].set_title('KL Divergence (Temperature-Scaled)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    add_event_markers(axes[0, 2], steps, sfa_merge_steps, checkpoint_steps)

    # Plot 4: L1 Distance (Router Divergence)
    # Check if router metrics are present (router might be frozen)
    has_router_metrics = any(d > 0 for d in l1_dists)

    if has_router_metrics:
        axes[1, 0].plot(steps, l1_dists, 'purple', linewidth=2, marker='o', markersize=3)
        axes[1, 0].axhline(y=0.29, color='green', linestyle='--', label='Target ~0.29', alpha=0.7)
        axes[1, 0].axhline(y=0.20, color='orange', linestyle='--', label='Warning threshold', alpha=0.5)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('L1 Distance')
        axes[1, 0].set_title('L1 Distance (Router Divergence)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        add_event_markers(axes[1, 0], steps, sfa_merge_steps, checkpoint_steps)
    else:
        axes[1, 0].text(0.5, 0.5, 'Router Frozen\n(No Divergence Metrics)',
                       ha='center', va='center', fontsize=12, color='gray',
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('L1 Distance (Router Divergence) - N/A')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('L1 Distance')

    # Plot 5: Routing Entropy Deviation (Student - Teacher)
    has_entropy_metrics = any(e > 0 for e in s_entropies)

    if has_entropy_metrics:
        # Calculate entropy deviation (student - teacher)
        entropy_deviations = [s - t for s, t in zip(s_entropies, t_entropies)]
        mean_deviation = np.mean(entropy_deviations)
        std_deviation = np.std(entropy_deviations)

        # Plot deviation over time
        axes[1, 1].plot(steps, entropy_deviations, 'purple', linewidth=2, marker='o', markersize=2)

        # Add ±1 and ±2 std dev reference bands
        axes[1, 1].axhline(y=mean_deviation, color='black', linestyle='-', linewidth=1, label=f'Mean ({mean_deviation:+.4f})', alpha=0.7)
        axes[1, 1].axhline(y=mean_deviation + std_deviation, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 1].axhline(y=mean_deviation - std_deviation, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 1].fill_between(steps,
                                mean_deviation - std_deviation,
                                mean_deviation + std_deviation,
                                alpha=0.15, color='orange',
                                label=f'±1σ ({std_deviation:.4f})')

        # Zero line for reference
        axes[1, 1].axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Entropy Deviation (bits)')
        axes[1, 1].set_title(f'Routing Entropy Deviation (μ={mean_deviation:+.4f}, σ={std_deviation:.4f})')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True, alpha=0.3)
        add_event_markers(axes[1, 1], steps, sfa_merge_steps, checkpoint_steps)
    else:
        axes[1, 1].text(0.5, 0.5, 'Router Frozen\n(No Entropy Metrics)',
                       ha='center', va='center', fontsize=12, color='gray',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Routing Entropy Deviation - N/A')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Deviation (bits)')

    # Plot 6: Learning Rate (enhanced with phase annotations)
    axes[1, 2].plot(steps, lrs, 'orange', linewidth=2, marker='o', markersize=3)

    # Add phase transition markers
    if phases['warmup_end']:
        axes[1, 2].axvline(x=phases['warmup_end'], color='blue', linestyle='--',
                          label=f'Warmup end (step {phases["warmup_end"]})', alpha=0.6)
    if phases['stable_end']:
        axes[1, 2].axvline(x=phases['stable_end'], color='green', linestyle='--',
                          label=f'Stable end (step {phases["stable_end"]})', alpha=0.6)

    # Add peak LR reference
    peak_lr = max(lrs)
    axes[1, 2].axhline(y=peak_lr, color='red', linestyle=':', label=f'Peak LR ({peak_lr:.2e})', alpha=0.4)

    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    add_event_markers(axes[1, 2], steps, sfa_merge_steps, checkpoint_steps)

    # Row 3: Gradient norm and advanced scheduler metrics
    if has_grad_norm or has_advanced:
        row_idx = 2

        # Plot 7: Gradient Norm
        if has_grad_norm:
            valid_grad_norms = [(s, g) for s, g in zip(steps, grad_norms) if g is not None]
            if valid_grad_norms:
                grad_steps, grad_vals = zip(*valid_grad_norms)
                axes[row_idx, 0].plot(grad_steps, grad_vals, 'darkgreen', linewidth=2, marker='o', markersize=3, label='Regular steps')

                # Overlay gradient clipping events (pre-clip norms)
                if clip_events:
                    # Filter out events without step numbers (old format compatibility)
                    valid_clips = [(e[0], e[1], e[2]) for e in clip_events if e[0] is not None]

                    if valid_clips:
                        clip_steps = [e[0] for e in valid_clips]
                        clip_norms = [e[1] for e in valid_clips]
                        clip_threshold = valid_clips[0][2]

                        # Plot clipping events as red stars (pre-clip values)
                        axes[row_idx, 0].scatter(clip_steps, clip_norms, color='red', marker='*', s=150,
                                                label=f'Clipping events (n={len(valid_clips)})', zorder=5,
                                                edgecolors='darkred', linewidths=1.5)

                        # Threshold line
                        axes[row_idx, 0].axhline(y=clip_threshold, color='red', linestyle='--',
                                                label=f'Clip threshold ({clip_threshold})', alpha=0.7)
                    else:
                        # No valid step numbers, just show threshold
                        clip_threshold = clip_events[0][2] if clip_events else 1.0
                        axes[row_idx, 0].axhline(y=clip_threshold, color='red', linestyle='--',
                                                label=f'Clip threshold ({clip_threshold})', alpha=0.7)
                else:
                    axes[row_idx, 0].axhline(y=1.0, color='red', linestyle='--', label='Clip threshold (1.0)', alpha=0.7)

                axes[row_idx, 0].set_xlabel('Step')
                axes[row_idx, 0].set_ylabel('Gradient Norm')
                axes[row_idx, 0].set_title('Gradient Norm (pre-clipping)')
                axes[row_idx, 0].legend()
                axes[row_idx, 0].grid(True, alpha=0.3)
                add_event_markers(axes[row_idx, 0], steps, sfa_merge_steps, checkpoint_steps)
        else:
            axes[row_idx, 0].axis('off')

        # Plot 8: Temperature Annealing
        if has_temperature:
            valid_temps = [(s, t) for s, t in zip(steps, temperatures) if t is not None]
            if valid_temps:
                temp_steps, temp_vals = zip(*valid_temps)
                axes[row_idx, 1].plot(temp_steps, temp_vals, 'darkred', linewidth=2, marker='o', markersize=3)
                axes[row_idx, 1].set_xlabel('Step')
                axes[row_idx, 1].set_ylabel('Temperature')
                axes[row_idx, 1].set_title('KL Distillation Temperature')
                axes[row_idx, 1].grid(True, alpha=0.3)
                add_event_markers(axes[row_idx, 1], steps, sfa_merge_steps, checkpoint_steps)

                # Add annotation for temperature values
                initial_temp = temp_vals[0]
                final_temp = temp_vals[-1]
                if initial_temp != final_temp:
                    axes[row_idx, 1].text(0.05, 0.95, f'T: {initial_temp:.1f} → {final_temp:.1f}',
                                         transform=axes[row_idx, 1].transAxes,
                                         fontsize=10, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            axes[row_idx, 1].axis('off')

        # Plot 9: KL Weight Schedule
        if has_kl_weight:
            valid_weights = [(s, w) for s, w in zip(steps, kl_weights) if w is not None]
            if valid_weights:
                weight_steps, weight_vals = zip(*valid_weights)
                axes[row_idx, 2].plot(weight_steps, weight_vals, 'darkblue', linewidth=2, marker='o', markersize=3)
                axes[row_idx, 2].set_xlabel('Step')
                axes[row_idx, 2].set_ylabel('KL Weight')
                axes[row_idx, 2].set_title('KL Loss Weight Schedule')
                axes[row_idx, 2].set_ylim([-0.05, 1.1])
                axes[row_idx, 2].grid(True, alpha=0.3)
                add_event_markers(axes[row_idx, 2], steps, sfa_merge_steps, checkpoint_steps)

                # Add warmup annotation if weight ramps up
                if weight_vals[0] < 0.5 and weight_vals[-1] > 0.9:
                    axes[row_idx, 2].text(0.05, 0.95, 'KL Weight Warmup',
                                         transform=axes[row_idx, 2].transAxes,
                                         fontsize=10, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        else:
            axes[row_idx, 2].axis('off')

    # Row 4: Advanced scheduler summary (if using advanced scheduler)
    if has_advanced:
        # Plot 10: Gradient Norm Distribution
        if has_grad_norm:
            valid_grad_norms_list = [g for g in grad_norms if g is not None]
            if valid_grad_norms_list:
                axes[3, 0].hist(valid_grad_norms_list, bins=30, color='darkgreen', alpha=0.7, edgecolor='black')
                axes[3, 0].axvline(x=1.0, color='red', linestyle='--', label='Clip threshold', alpha=0.7)
                mean_grad = np.mean(valid_grad_norms_list)
                axes[3, 0].axvline(x=mean_grad, color='blue', linestyle='-', label=f'Mean={mean_grad:.3f}', alpha=0.7)
                axes[3, 0].set_xlabel('Gradient Norm')
                axes[3, 0].set_ylabel('Frequency')
                clipped_count = sum(1 for g in valid_grad_norms_list if g > 1.0)
                clip_pct = 100 * clipped_count / len(valid_grad_norms_list)
                axes[3, 0].set_title(f'Gradient Norm Distribution (Clipped: {clip_pct:.1f}%)')
                axes[3, 0].legend()
                axes[3, 0].grid(True, alpha=0.3)
        else:
            axes[3, 0].axis('off')

        # Plot 11: Hyperparameter Schedule Overview
        ax_overview = axes[3, 1]
        ax_overview.axis('off')

        # Create summary of schedule
        summary_lines = ["Advanced Scheduler Summary:\n"]

        # LR phases
        if phases['warmup_end']:
            warmup_pct = 100 * phases['warmup_end'] / max(steps)
            summary_lines.append(f"Warmup: steps 0-{phases['warmup_end']} ({warmup_pct:.1f}%)")

        if phases['stable_end']:
            stable_start = phases['warmup_end'] or 0
            stable_pct = 100 * (phases['stable_end'] - stable_start) / max(steps)
            summary_lines.append(f"Stable: steps {stable_start}-{phases['stable_end']} ({stable_pct:.1f}%)")

        if phases['stable_end']:
            decay_pct = 100 * (max(steps) - phases['stable_end']) / max(steps)
            summary_lines.append(f"Decay: steps {phases['stable_end']}-{max(steps)} ({decay_pct:.1f}%)")

        peak_lr = max(lrs)
        final_lr = lrs[-1]
        final_lr_ratio = final_lr / peak_lr if peak_lr > 0 else 0
        summary_lines.append(f"\nPeak LR: {peak_lr:.2e}")
        summary_lines.append(f"Final LR: {final_lr:.2e} ({final_lr_ratio:.1%} of peak)")

        if has_temperature:
            valid_temps = [t for t in temperatures if t is not None]
            if valid_temps:
                summary_lines.append(f"\nTemperature: {valid_temps[0]:.1f} → {valid_temps[-1]:.1f}")

        if has_kl_weight:
            valid_weights = [w for w in kl_weights if w is not None]
            if valid_weights and valid_weights[0] < 0.5:
                summary_lines.append("KL Weight Warmup: enabled")

        # Detect router bias freeze (LR becomes 0 at some point)
        # This would require bias_lr in metrics, which we don't log separately yet
        # For now, estimate based on expected 90% freeze point
        if max(steps) > 100:
            expected_freeze = int(0.9 * max(steps))
            summary_lines.append(f"\nExpected bias freeze: ~step {expected_freeze}")

        summary_text = "\n".join(summary_lines)
        ax_overview.text(0.1, 0.5, summary_text, fontsize=11,
                        verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Plot 12: Training Summary
        axes[3, 2].axis('off')
        summary_text = (
            f"Training Summary:\n\n"
            f"Total Steps: {len(steps)}\n"
            f"Step Range: {min(steps)}-{max(steps)}\n\n"
            f"Final Metrics:\n"
            f"  Loss: {losses[-1]:.4f}\n"
            f"  LM Loss: {lm_losses[-1]:.4f}\n"
            f"  KL Loss: {kl_losses[-1]:.4f}\n"
            f"  L1 Dist: {l1_dists[-1]:.4f}\n"
            f"  S Entropy: {s_entropies[-1]:.4f}\n"
            f"  T Entropy: {t_entropies[-1]:.4f}\n\n"
        )

        if has_grad_norm:
            valid_grad_norms_list = [g for g in grad_norms if g is not None]
            if valid_grad_norms_list:
                mean_grad = np.mean(valid_grad_norms_list)
                clipped_count = sum(1 for g in valid_grad_norms_list if g > 1.0)
                clip_pct = 100 * clipped_count / len(valid_grad_norms_list)
                summary_text += (
                    f"Gradient Stats:\n"
                    f"  Mean: {mean_grad:.4f}\n"
                    f"  Max: {max(valid_grad_norms_list):.4f}\n"
                    f"  Clipped: {clip_pct:.1f}%"
                )

        axes[3, 2].text(0.1, 0.5, summary_text, fontsize=11,
                       verticalalignment='center', family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    print(f"Steps plotted: {min(steps)} to {max(steps)} ({len(steps)} total)")

    # Print detected phases
    if phases['warmup_end'] or phases['stable_end']:
        print("\nDetected Schedule Phases:")
        if phases['warmup_end']:
            print(f"  Warmup phase: steps 0-{phases['warmup_end']}")
        if phases['stable_end']:
            warmup_end = phases['warmup_end'] or 0
            print(f"  Stable phase: steps {warmup_end}-{phases['stable_end']}")
        if phases['stable_end']:
            print(f"  Decay phase: steps {phases['stable_end']}-{max(steps)}")


def main():
    parser = argparse.ArgumentParser(description='Plot Stage-0 training metrics')
    parser.add_argument('--log', type=str, help='Path to training log file')
    parser.add_argument('--data', type=str, help='Path to text file with step lines')
    parser.add_argument('--jsonl', type=str, help='Path to metrics.jsonl file (comprehensive)')
    parser.add_argument('--output', type=str, default='training_metrics.png',
                       help='Output PNG path (default: training_metrics.png)')
    parser.add_argument('--title', type=str, default='',
                       help='Optional title suffix (e.g., " (Advanced Scheduler)")')

    args = parser.parse_args()

    # Load data from appropriate source
    clip_events = []  # Default: no clipping events
    sfa_merge_steps = []  # Default: no SFA merge events
    checkpoint_steps = []  # Default: no checkpoint events

    if args.jsonl:
        print(f"Loading data from JSONL file: {args.jsonl}")
        steps, all_metrics = load_data_from_jsonl(args.jsonl)
    elif args.log:
        print(f"Loading data from log file: {args.log}")
        steps, all_metrics, clip_events, sfa_merge_steps, checkpoint_steps = load_data_from_log(args.log)
        if clip_events:
            print(f"  Found {len(clip_events)} gradient clipping events")
        if sfa_merge_steps:
            print(f"  Found {len(sfa_merge_steps)} SFA merge events")
        if checkpoint_steps:
            print(f"  Found {len(checkpoint_steps)} checkpoint saves")
    elif args.data:
        print(f"Loading data from text file: {args.data}")
        with open(args.data, 'r') as f:
            text = f.read()
        steps, all_metrics = load_data_from_text(text)
    else:
        print("No log, data, or jsonl file specified. Paste your step lines below (Ctrl+D when done):")
        text = sys.stdin.read()
        steps, all_metrics = load_data_from_text(text)

    if not steps:
        print("Error: No valid step data found!")
        return 1

    # Create plot
    plot_metrics(steps, all_metrics, args.output, args.title,
                clip_events=clip_events,
                sfa_merge_steps=sfa_merge_steps,
                checkpoint_steps=checkpoint_steps)

    # Print summary stats
    print("\nSummary:")
    print(f"  Total loss:    {all_metrics[0]['loss']:.4f} → {all_metrics[-1]['loss']:.4f}")
    lm_loss_key = 'lm_loss' if 'lm_loss' in all_metrics[0] else 'lm'
    kl_loss_key = 'kl_loss' if 'kl_loss' in all_metrics[0] else 'kl'
    l1_dist_key = 'avg_l1_distance' if 'avg_l1_distance' in all_metrics[0] else 'l1_dist'

    print(f"  LM loss:       {all_metrics[0][lm_loss_key]:.4f} → {all_metrics[-1][lm_loss_key]:.4f}")
    print(f"  KL divergence: {all_metrics[0][kl_loss_key]:.4f} → {all_metrics[-1][kl_loss_key]:.4f}")
    print(f"  L1 distance:   {all_metrics[0][l1_dist_key]:.4f} → {all_metrics[-1][l1_dist_key]:.4f}")

    # Print advanced scheduler metrics if available
    if 'temperature' in all_metrics[0] and all_metrics[0]['temperature'] is not None:
        print(f"  Temperature:   {all_metrics[0]['temperature']:.2f} → {all_metrics[-1]['temperature']:.2f}")
    if 'kl_weight' in all_metrics[0] and all_metrics[0]['kl_weight'] is not None:
        print(f"  KL weight:     {all_metrics[0]['kl_weight']:.2f} → {all_metrics[-1]['kl_weight']:.2f}")

    return 0


if __name__ == '__main__':
    exit(main())
