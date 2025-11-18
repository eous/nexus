#!/usr/bin/env python3
"""
Learning rate and training schedulers for GPT-OSS Stage-0.

Adapted from DeepSeek-V3's advanced scheduling techniques:
- Multi-phase LR schedule (warmup → stable → cosine decay)
- Router bias scheduling (slow updates → freeze)
- Temperature annealing for KL distillation
- KL weight warmup

Key improvements over standard schedules:
1. Non-zero final LR (maintains learning capacity)
2. Stable phase prevents premature convergence
3. Slow router bias updates improve load balancing
4. Late bias freeze stabilizes routing decisions
5. Temperature annealing improves distillation
"""

import math
from typing import Optional, Dict, Any, List
import torch
from torch.optim import Optimizer
import warnings


class GPTOSSLRScheduler:
    """
    Learning rate scheduler with warmup, stable, and cosine decay phases.

    Based on DeepSeek-V3's multi-stage schedule, optimized for MoE training:
    - Phase 1: Linear warmup to peak LR
    - Phase 2: Stable phase at peak LR (prevents premature convergence)
    - Phase 3: Cosine decay to final LR (NOT zero - maintains learning capacity)
    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int,
                 stable_steps: int,
                 decay_steps: int,
                 peak_lr: float,
                 final_lr: Optional[float] = None,
                 min_lr_ratio: float = 0.1):
        """
        Initialize LR scheduler.

        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of warmup steps
            stable_steps: Number of steps at peak LR
            decay_steps: Number of decay steps
            peak_lr: Peak learning rate after warmup
            final_lr: Final learning rate after decay (default: peak_lr * 0.1)
            min_lr_ratio: Minimum LR ratio during warmup
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.peak_lr = peak_lr
        self.final_lr = final_lr if final_lr is not None else peak_lr * 0.1
        self.min_lr_ratio = min_lr_ratio
        self.current_step = 0

        # Initialize optimizer LR
        self._update_lr(self.get_lr(0))

    def get_lr(self, step: Optional[int] = None) -> float:
        """Calculate learning rate for a given step."""
        if step is None:
            step = self.current_step

        # Phase 1: Warmup
        if step < self.warmup_steps:
            warmup_lr = self.peak_lr * step / max(1, self.warmup_steps)
            min_lr = self.peak_lr * self.min_lr_ratio
            return max(warmup_lr, min_lr)

        # Phase 2: Stable
        if step < self.warmup_steps + self.stable_steps:
            return self.peak_lr

        # Phase 3: Cosine decay
        decay_step = step - self.warmup_steps - self.stable_steps
        if decay_step >= self.decay_steps:
            return self.final_lr

        progress = decay_step / self.decay_steps
        return self.final_lr + (self.peak_lr - self.final_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def _update_lr(self, lr: float):
        """Update learning rate in all parameter groups."""
        for param_group in self.optimizer.param_groups:
            if isinstance(param_group['lr'], torch.Tensor):
                param_group['lr'].fill_(lr)
            else:
                param_group['lr'] = lr

    def step(self) -> float:
        """Update learning rate for current step."""
        self.current_step += 1
        lr = self.get_lr()
        self._update_lr(lr)
        return lr

    def get_last_lr(self) -> float:
        """Get the last computed learning rate."""
        return self.get_lr()

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'stable_steps': self.stable_steps,
            'decay_steps': self.decay_steps,
            'peak_lr': self.peak_lr,
            'final_lr': self.final_lr,
            'min_lr_ratio': self.min_lr_ratio
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.stable_steps = state_dict['stable_steps']
        self.decay_steps = state_dict['decay_steps']
        self.peak_lr = state_dict['peak_lr']
        self.final_lr = state_dict['final_lr']
        self.min_lr_ratio = state_dict.get('min_lr_ratio', 0.1)
        self._update_lr(self.get_lr())


class GPTOSSTrainingScheduler:
    """
    Comprehensive training scheduler for GPT-OSS Stage-0 that coordinates:

    1. Learning rate: warmup → stable → cosine decay
    2. Router bias updates: slow (LR×0.001) → freeze at 90%
    3. Temperature: high (soft) → low (sharp) annealing
    4. KL weight: gradual warmup during LR warmup

    Based on DeepSeek-V3's advanced scheduling with adaptations for:
    - KL distillation from teacher model
    - MoE router stability
    - Single-GPU training constraints
    """

    def __init__(self,
                 model,
                 optimizer: Optimizer,
                 total_steps: int,
                 peak_lr: float,
                 warmup_ratio: float = 0.1,
                 stable_ratio: float = 0.2,
                 final_lr_ratio: float = 0.1,
                 bias_lr_multiplier: float = 0.001,
                 bias_freeze_ratio: float = 0.9,
                 initial_temperature: float = 4.0,
                 final_temperature: float = 2.0,
                 temperature_anneal_start_ratio: float = 0.3,
                 kl_weight_warmup: bool = True):
        """
        Initialize comprehensive training scheduler.

        Args:
            model: The student model
            optimizer: PyTorch optimizer (must have router_bias and other param groups)
            total_steps: Total training steps
            peak_lr: Peak learning rate
            warmup_ratio: Fraction of steps for warmup (default: 0.1 = 10%)
            stable_ratio: Fraction of steps for stable phase (default: 0.2 = 20%)
            final_lr_ratio: Final LR as fraction of peak (default: 0.1 = 10% of peak)
            bias_lr_multiplier: Router bias LR multiplier (default: 0.001 = 0.1% of main LR)
            bias_freeze_ratio: Freeze biases after this fraction of training (default: 0.9 = 90%)
            initial_temperature: Starting temperature for KL distillation (default: 4.0)
            final_temperature: Final temperature (default: 2.0)
            temperature_anneal_start_ratio: Start annealing after this fraction (default: 0.3 = 30%)
            kl_weight_warmup: Gradually increase KL weight during warmup (default: True)
        """
        self.model = model
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.peak_lr = peak_lr

        # Calculate phase lengths
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.stable_steps = int(stable_ratio * total_steps)
        self.decay_steps = total_steps - self.warmup_steps - self.stable_steps
        self.final_lr = peak_lr * final_lr_ratio

        # Router bias scheduling
        self.bias_lr_multiplier = bias_lr_multiplier
        if bias_freeze_ratio is not None:
            self.bias_freeze_step = int(bias_freeze_ratio * total_steps)
            self.bias_frozen = False
        else:
            # No bias freezing (router already frozen or not used)
            self.bias_freeze_step = None
            self.bias_frozen = True

        # Temperature annealing
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_anneal_start = int(temperature_anneal_start_ratio * total_steps)

        # KL weight scheduling
        self.kl_weight_warmup = kl_weight_warmup

        # LR scheduler
        self.lr_scheduler = GPTOSSLRScheduler(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            stable_steps=self.stable_steps,
            decay_steps=self.decay_steps,
            peak_lr=peak_lr,
            final_lr=self.final_lr
        )

        # Track which parameter group is router biases (if separated)
        self._find_bias_param_group()

        self.current_step = 0

    def _find_bias_param_group(self):
        """Identify which parameter group contains router biases."""
        self.bias_param_group_idx = None

        for idx, param_group in enumerate(self.optimizer.param_groups):
            # Check if this group has the 'name' key identifying it as router biases
            if param_group.get('name') == 'router_bias':
                self.bias_param_group_idx = idx
                break

    def get_temperature(self, step: Optional[int] = None) -> float:
        """
        Get temperature for KL distillation at given step.

        Implements annealing: high (soft) → low (sharp)
        - Early: T=4.0 (softer targets, easier to learn)
        - Late: T=2.0 (sharper targets, more precise)
        """
        if step is None:
            step = self.current_step

        if step < self.temperature_anneal_start:
            return self.initial_temperature

        # Linear annealing from initial to final
        progress = (step - self.temperature_anneal_start) / (self.total_steps - self.temperature_anneal_start)
        progress = min(1.0, progress)

        return self.initial_temperature - (self.initial_temperature - self.final_temperature) * progress

    def get_kl_weight(self, step: Optional[int] = None) -> float:
        """
        Get KL loss weight at given step.

        If warmup enabled:
        - Ramp from 0 to 1.0 during warmup
        - Helps model establish basic patterns before strong distillation

        Otherwise: constant 1.0
        """
        if step is None:
            step = self.current_step

        if not self.kl_weight_warmup or step >= self.warmup_steps:
            return 1.0

        # Linear warmup from 0 to 1.0
        return step / max(1, self.warmup_steps)

    def freeze_router_biases(self):
        """
        Freeze router biases to stabilize routing decisions.

        Called automatically at bias_freeze_step (default: 90% of training).
        """
        if self.bias_frozen:
            return

        frozen_count = 0
        for name, param in self.model.named_parameters():
            if 'mlp.router.bias' in name and param.requires_grad:
                param.requires_grad = False
                frozen_count += 1

        self.bias_frozen = True

        # Remove router bias parameters from optimizer
        if self.bias_param_group_idx is not None:
            # Mark the param group as having 0 LR (effectively frozen)
            self.optimizer.param_groups[self.bias_param_group_idx]['lr'] = 0.0

        print(f"\n{'='*80}")
        print(f"ROUTER BIAS FREEZE @ Step {self.current_step}")
        print(f"{'='*80}")
        print(f"  Frozen {frozen_count} router bias parameters")
        print(f"  Rationale: Stabilize routing decisions for final convergence")
        print(f"  Router weights continue training to refine expert selection")
        print(f"{'='*80}\n")

    def get_last_lr(self) -> List[float]:
        """
        Get the last computed learning rate (for compatibility with PyTorch schedulers).

        Returns:
            List with single LR value (format expected by training scripts)
        """
        return [self.lr_scheduler.get_last_lr()]

    def step(self) -> Dict[str, float]:
        """
        Update all scheduled parameters for current training step.

        Returns:
            Dictionary with current values: lr, temperature, kl_weight, bias_frozen
        """
        self.current_step += 1

        # Update main learning rate
        lr = self.lr_scheduler.step()

        # Update router bias LR (if we have separate param groups)
        if self.bias_param_group_idx is not None and not self.bias_frozen:
            bias_lr = lr * self.bias_lr_multiplier
            if isinstance(self.optimizer.param_groups[self.bias_param_group_idx]['lr'], torch.Tensor):
                self.optimizer.param_groups[self.bias_param_group_idx]['lr'].fill_(bias_lr)
            else:
                self.optimizer.param_groups[self.bias_param_group_idx]['lr'] = bias_lr

        # Freeze router biases if we hit the threshold
        if self.bias_freeze_step is not None and self.current_step == self.bias_freeze_step:
            self.freeze_router_biases()

        # Get temperature and KL weight
        temperature = self.get_temperature()
        kl_weight = self.get_kl_weight()

        return {
            'lr': lr,
            'bias_lr': lr * self.bias_lr_multiplier if not self.bias_frozen else 0.0,
            'temperature': temperature,
            'kl_weight': kl_weight,
            'bias_frozen': self.bias_frozen,
            'step': self.current_step,
            'phase': self._get_phase_name()
        }

    def _get_phase_name(self) -> str:
        """Get current training phase name."""
        if self.current_step < self.warmup_steps:
            return 'warmup'
        elif self.current_step < self.warmup_steps + self.stable_steps:
            return 'stable'
        else:
            return 'decay'

    def get_schedule_info(self) -> Dict[str, Any]:
        """
        Get comprehensive schedule information for logging.

        Returns summary of all schedule parameters and milestones.
        """
        return {
            'total_steps': self.total_steps,
            'warmup_steps': self.warmup_steps,
            'stable_steps': self.stable_steps,
            'decay_steps': self.decay_steps,
            'peak_lr': self.peak_lr,
            'final_lr': self.final_lr,
            'bias_lr_multiplier': self.bias_lr_multiplier,
            'bias_freeze_step': self.bias_freeze_step,
            'initial_temperature': self.initial_temperature,
            'final_temperature': self.final_temperature,
            'temperature_anneal_start': self.temperature_anneal_start,
            'kl_weight_warmup': self.kl_weight_warmup
        }

    def state_dict(self) -> Dict[str, Any]:
        """Return full scheduler state for checkpointing."""
        return {
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'current_step': self.current_step,
            'bias_frozen': self.bias_frozen,
            'schedule_info': self.get_schedule_info()
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load full scheduler state from checkpoint."""
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.current_step = state_dict['current_step']
        self.bias_frozen = state_dict.get('bias_frozen', False)

        # Reapply bias freeze if we were frozen
        if self.bias_frozen:
            self.freeze_router_biases()


def create_optimizer_with_bias_groups(model,
                                      learning_rate: float,
                                      bias_lr_multiplier: float = 0.001,
                                      verbose: bool = True) -> Optimizer:
    """
    Create optimizer with separate parameter groups for router biases.

    This enables differential learning rates:
    - Main parameters: full learning rate
    - Router biases: learning_rate × bias_lr_multiplier (default: 0.1% of main)

    Args:
        model: The model with trainable parameters
        learning_rate: Base learning rate for main parameters
        bias_lr_multiplier: Multiplier for router bias learning rate
        verbose: Whether to print parameter group info (default: True)

    Returns:
        AdamW optimizer with separate parameter groups
    """
    from torch.optim import AdamW

    router_bias_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'mlp.router.bias' in name:
                router_bias_params.append(param)
            else:
                other_params.append(param)

    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': other_params,
            'lr': learning_rate,
            'name': 'main'
        }
    ]

    if router_bias_params:
        param_groups.append({
            'params': router_bias_params,
            'lr': learning_rate * bias_lr_multiplier,
            'name': 'router_bias'
        })

    optimizer = AdamW(param_groups)

    # Print parameter group info (optional, for DDP main process only)
    if verbose:
        print("\nOptimizer Parameter Groups:")
        print(f"  Main parameters: {sum(p.numel() for p in other_params):,} params @ LR={learning_rate:.2e}")
        if router_bias_params:
            print(f"  Router biases: {sum(p.numel() for p in router_bias_params):,} params @ LR={learning_rate * bias_lr_multiplier:.2e} (×{bias_lr_multiplier})")
        else:
            print(f"  Router biases: None found (model may not have router biases)")

    return optimizer


def print_schedule_summary(scheduler: GPTOSSTrainingScheduler):
    """
    Print comprehensive summary of training schedule.

    Args:
        scheduler: GPTOSSTrainingScheduler instance
    """
    info = scheduler.get_schedule_info()

    print("\n" + "="*80)
    print("TRAINING SCHEDULE SUMMARY (DeepSeek-V3 Style)")
    print("="*80)

    print("\nLearning Rate Schedule:")
    print(f"  Phase 1 - Warmup:  Steps 0-{info['warmup_steps']:,} "
          f"({100*info['warmup_steps']/info['total_steps']:.1f}%)")
    print(f"    LR ramps from ~{info['peak_lr']*0.1:.2e} to {info['peak_lr']:.2e}")

    print(f"  Phase 2 - Stable:  Steps {info['warmup_steps']:,}-{info['warmup_steps']+info['stable_steps']:,} "
          f"({100*info['stable_steps']/info['total_steps']:.1f}%)")
    print(f"    LR constant at {info['peak_lr']:.2e}")

    print(f"  Phase 3 - Decay:   Steps {info['warmup_steps']+info['stable_steps']:,}-{info['total_steps']:,} "
          f"({100*info['decay_steps']/info['total_steps']:.1f}%)")
    print(f"    LR cosine decay from {info['peak_lr']:.2e} to {info['final_lr']:.2e}")

    print(f"\n  Final LR ratio: {info['final_lr']/info['peak_lr']:.1%} of peak (NOT zero!)")

    print("\nRouter Bias Schedule:")
    if info['bias_freeze_step'] is not None:
        print(f"  Phase 1: Steps 0-{info['bias_freeze_step']:,} "
              f"({100*info['bias_freeze_step']/info['total_steps']:.1f}%)")
        print(f"    Slow updates @ LR×{info['bias_lr_multiplier']} (prevents drift)")
        print(f"  Phase 2: Steps {info['bias_freeze_step']:,}-{info['total_steps']:,} "
              f"({100*(info['total_steps']-info['bias_freeze_step'])/info['total_steps']:.1f}%)")
        print(f"    FROZEN (stabilizes routing for final convergence)")
    else:
        print(f"  Router FROZEN (--freeze-router enabled)")
        print(f"    Training shared expert only, router not updated")

    print("\nTemperature Schedule (KL Distillation):")
    print(f"  Phase 1: Steps 0-{info['temperature_anneal_start']:,} "
          f"({100*info['temperature_anneal_start']/info['total_steps']:.1f}%)")
    print(f"    Constant @ T={info['initial_temperature']:.1f} (soft targets)")
    print(f"  Phase 2: Steps {info['temperature_anneal_start']:,}-{info['total_steps']:,} "
          f"({100*(info['total_steps']-info['temperature_anneal_start'])/info['total_steps']:.1f}%)")
    print(f"    Linear anneal from T={info['initial_temperature']:.1f} to T={info['final_temperature']:.1f} (sharper targets)")

    if info['kl_weight_warmup']:
        print("\nKL Weight Schedule:")
        print(f"  Warmup: 0.0 → 1.0 over steps 0-{info['warmup_steps']:,}")
        print(f"  Stable: 1.0 for remaining steps")

    print("\n" + "="*80)
    print()
