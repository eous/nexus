#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
Runtime Sample Verification for Offline Training

Verifies that:
1. Precomputed samples match what was actually saved during teacher inference
2. DDP interleaving is correct (no duplicates across GPUs)
3. Sample ordering is deterministic and correct

Usage:
    from sample_verification import verify_sample_integrity, track_ddp_samples
"""

import hashlib
import torch
import numpy as np
from typing import Dict, Set, Optional


def compute_input_ids_hash(input_ids: torch.Tensor) -> str:
    """
    Compute a deterministic hash of input_ids for verification.

    Args:
        input_ids: Tensor of shape (batch_size, seq_len)

    Returns:
        Hex string hash (first 16 chars for brevity)
    """
    # Convert to numpy, ensure deterministic ordering
    input_ids_np = input_ids.cpu().numpy()

    # Use SHA256 for cryptographic strength
    hash_obj = hashlib.sha256()
    hash_obj.update(input_ids_np.tobytes())

    return hash_obj.hexdigest()[:16]  # First 16 chars sufficient for uniqueness


def verify_sample_integrity(
    precomputed_input_ids: torch.Tensor,
    saved_hash: str,
    step: int,
    rank: int = 0,
    verbose: bool = True
) -> bool:
    """
    Verify that precomputed input_ids match their saved hash.

    Args:
        precomputed_input_ids: Input IDs loaded from precomputed file
        saved_hash: Hash saved during precomputation (required)
        step: Global step number
        rank: Process rank (for logging)
        verbose: Print verification results

    Returns:
        True if hash matches, False if mismatch
    """
    # Compute current hash
    current_hash = compute_input_ids_hash(precomputed_input_ids)

    if current_hash != saved_hash:
        print(f"  ✗✗✗ SAMPLE INTEGRITY VERIFICATION FAILED ✗✗✗")
        print(f"  Rank {rank}, Step {step}")
        print(f"  Expected hash: {saved_hash}")
        print(f"  Actual hash:   {current_hash}")
        print(f"  This indicates the precomputed data is corrupted or mismatched!")
        return False

    if verbose and step < 3:  # Log first few steps
        print(f"  ✓ [Rank {rank}, Step {step}] Sample integrity verified (hash: {current_hash})")

    return True


class DDPSampleTracker:
    """
    Track samples across DDP processes to detect duplicates or gaps.

    Usage:
        tracker = DDPSampleTracker(num_processes=2)
        tracker.record_sample(step=0, rank=0, input_ids=batch['input_ids'])
        tracker.verify_no_duplicates()  # Call periodically
    """

    def __init__(self, num_processes: int, check_interval: int = 100):
        """
        Initialize tracker.

        Args:
            num_processes: Number of DDP processes
            check_interval: Check for duplicates every N steps
        """
        self.num_processes = num_processes
        self.check_interval = check_interval

        # Track hashes seen by each rank: rank -> set of hashes
        self.samples_by_rank: Dict[int, Set[str]] = {
            rank: set() for rank in range(num_processes)
        }

        # Track step -> rank mapping to verify interleaving pattern
        self.step_to_rank: Dict[int, int] = {}

        self.steps_recorded = 0

    def record_sample(
        self,
        step: int,
        rank: int,
        input_ids: torch.Tensor
    ):
        """
        Record a sample processed by a specific rank.

        Args:
            step: Global step number
            rank: Process rank
            input_ids: Input IDs tensor (batch_size, seq_len)
        """
        # Compute hash
        sample_hash = compute_input_ids_hash(input_ids)

        # Record for this rank
        self.samples_by_rank[rank].add(sample_hash)

        # Record step-to-rank mapping
        self.step_to_rank[step] = rank

        self.steps_recorded += 1

    def verify_no_duplicates(self, verbose: bool = True) -> bool:
        """
        Verify no sample has been processed by multiple ranks.

        Returns:
            True if no duplicates found
        """
        # Get all samples from all ranks
        all_samples = []
        for rank, samples in self.samples_by_rank.items():
            all_samples.extend([(sample, rank) for sample in samples])

        # Check for duplicates
        seen_hashes = {}
        duplicates = []

        for sample_hash, rank in all_samples:
            if sample_hash in seen_hashes:
                duplicates.append((sample_hash, seen_hashes[sample_hash], rank))
            else:
                seen_hashes[sample_hash] = rank

        if duplicates:
            print(f"\n  ✗✗✗ DUPLICATE SAMPLES DETECTED ACROSS RANKS ✗✗✗")
            print(f"  Found {len(duplicates)} duplicate(s):")
            for sample_hash, rank1, rank2 in duplicates[:5]:  # Show first 5
                print(f"    Hash {sample_hash}: processed by rank {rank1} AND rank {rank2}")
            print(f"  This indicates DDP interleaving is broken!")
            return False

        if verbose:
            total_unique = len(seen_hashes)
            print(f"  ✓ No duplicates found across {self.num_processes} ranks")
            print(f"    Total unique samples: {total_unique}")

        return True

    def verify_interleaving_pattern(self, verbose: bool = True) -> bool:
        """
        Verify steps are interleaved correctly across ranks.

        Expected pattern with 2 processes:
            Step 0 → Rank 0
            Step 1 → Rank 1
            Step 2 → Rank 0
            Step 3 → Rank 1
            ...

        Returns:
            True if pattern is correct
        """
        if not self.step_to_rank:
            return True  # No data yet

        # Check pattern
        errors = []
        for step, actual_rank in sorted(self.step_to_rank.items()):
            expected_rank = step % self.num_processes

            if actual_rank != expected_rank:
                errors.append((step, expected_rank, actual_rank))

        if errors:
            print(f"\n  ✗✗✗ INTERLEAVING PATTERN INCORRECT ✗✗✗")
            print(f"  Found {len(errors)} mismatch(es):")
            for step, expected, actual in errors[:5]:
                print(f"    Step {step}: expected rank {expected}, got rank {actual}")
            print(f"  This indicates the DDP indexing formula is wrong!")
            return False

        if verbose:
            print(f"  ✓ Interleaving pattern correct for {len(self.step_to_rank)} steps")

        return True

    def should_check(self) -> bool:
        """Check if we should verify duplicates (every N steps)."""
        return self.steps_recorded % self.check_interval == 0


def track_ddp_samples(
    accelerator,
    precomputed_step: int,
    input_ids: torch.Tensor,
    tracker: Optional[DDPSampleTracker] = None
) -> bool:
    """
    Track samples across DDP processes using distributed communication.

    This is the DDP-aware version that uses all_gather to check for
    duplicates across processes in real-time.

    Args:
        accelerator: Accelerate accelerator object
        precomputed_step: Global step number
        input_ids: Input IDs for this batch
        tracker: Optional local tracker (for non-DDP fallback)

    Returns:
        True if no duplicates detected
    """
    if accelerator is None or not accelerator.num_processes > 1:
        # Non-DDP mode, use local tracker
        if tracker:
            tracker.record_sample(precomputed_step, 0, input_ids)
            return True
        return True

    # DDP mode: use distributed verification
    rank = accelerator.process_index
    num_processes = accelerator.num_processes

    # Compute hash of first sample in batch (sufficient for uniqueness)
    sample_hash = compute_input_ids_hash(input_ids[0:1])  # Just first sample

    # Convert hash to tensor for all_gather
    hash_bytes = sample_hash.encode('utf-8')
    hash_tensor = torch.tensor(
        [ord(c) for c in sample_hash],
        dtype=torch.long,
        device=accelerator.device
    )

    # Gather hashes from all processes
    gathered_hashes = accelerator.gather(hash_tensor.unsqueeze(0))

    # Check for duplicates (only on main process to avoid duplicate prints)
    if accelerator.is_main_process:
        # Convert back to strings
        hashes = []
        for i in range(num_processes):
            hash_tensor_i = gathered_hashes[i]
            hash_str = ''.join([chr(int(c)) for c in hash_tensor_i])
            hashes.append(hash_str)

        # Check for duplicates
        if len(hashes) != len(set(hashes)):
            print(f"\n  ✗✗✗ DUPLICATE DETECTED AT STEP {precomputed_step} ✗✗✗")
            for i, h in enumerate(hashes):
                print(f"    Rank {i}: {h}")
            return False

        # Verify interleaving pattern
        expected_rank = precomputed_step % num_processes
        if rank != expected_rank and precomputed_step < 10:  # Check first 10 steps
            print(f"  Warning: Step {precomputed_step} on rank {rank}, expected rank {expected_rank}")

    return True


# Example usage in train_offline.py:
"""
# At the top of train_offline.py:
from sample_verification import (
    verify_sample_integrity,
    DDPSampleTracker,
    track_ddp_samples
)

# After loading precomputed_loader:
tracker = DDPSampleTracker(
    num_processes=accelerator.num_processes,
    check_interval=100
) if args.verify_samples else None

# In the training loop, after loading precomputed batch:
if args.verify_samples:
    # Verify hash matches (if saved)
    saved_hash = precomputed_outputs.metadata.get('input_ids_hash')
    if not verify_sample_integrity(
        batch['input_ids'],
        saved_hash,
        step=precomputed_step,
        rank=accelerator.process_index,
        verbose=(precomputed_step < 3)
    ):
        raise RuntimeError(f"Sample integrity check failed at step {precomputed_step}")

    # Track for duplicate detection
    tracker.record_sample(
        step=precomputed_step,
        rank=accelerator.process_index,
        input_ids=batch['input_ids']
    )

    # Periodically verify no duplicates
    if tracker.should_check():
        if not tracker.verify_no_duplicates(verbose=True):
            raise RuntimeError("Duplicate samples detected across DDP ranks")
        if not tracker.verify_interleaving_pattern(verbose=True):
            raise RuntimeError("DDP interleaving pattern incorrect")
"""
