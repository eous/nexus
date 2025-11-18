#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
GPT-OSS Router Statistics Collection

Collect router probabilities for PCA-guided expert merging.

This script collects full router probability distributions (all 128 experts)
across all layers for a calibration dataset. It uses distribution-aware sampling
to ensure the selected samples provide good coverage of the token vocabulary.

Usage:
    python collect_router_probabilities.py \
        --model /mnt/models/gpt-oss-120b \
        --target-tokens 100000 \
        --output data/router_probs.npz
"""

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import interleave_datasets, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable tokenizers parallelism to avoid fork warnings with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_dataset_by_name(
    dataset_name: str,
    args,
) -> tuple:
    """
    Load dataset by name with support for nemotron-mixed interleaving.

    Args:
        dataset_name: Name of dataset or 'nemotron-mixed'
        args: Parsed arguments with local paths and fractions

    Returns:
        (dataset, dataset_type) tuple
    """
    if dataset_name == "nemotron-mixed":
        # Load individual nemotron datasets
        print(f"Loading nemotron-mixed with fractions: code={args.nemotron_code_fraction}, "
              f"math={args.nemotron_math_fraction}, tool={args.nemotron_tool_fraction}")

        datasets_to_load = []
        probabilities = []

        # Load code dataset
        if args.nemotron_code_fraction > 0:
            try:
                if Path(args.local_code_path).exists():
                    code_ds = load_from_disk(args.local_code_path).to_iterable_dataset()
                else:
                    code_ds = load_dataset("nvidia/HelpSteer2", split="train", streaming=True)
                datasets_to_load.append(code_ds)
                probabilities.append(args.nemotron_code_fraction)
                print(f"  Loaded nemotron-code (fraction: {args.nemotron_code_fraction})")
            except Exception as e:
                print(f"  WARNING: Could not load code dataset: {e}")

        # Load math dataset
        if args.nemotron_math_fraction > 0:
            try:
                if Path(args.local_math_path).exists():
                    math_ds = load_from_disk(args.local_math_path).to_iterable_dataset()
                else:
                    math_ds = load_dataset("nvidia/HelpSteer2", split="train", streaming=True)
                datasets_to_load.append(math_ds)
                probabilities.append(args.nemotron_math_fraction)
                print(f"  Loaded nemotron-math (fraction: {args.nemotron_math_fraction})")
            except Exception as e:
                print(f"  WARNING: Could not load math dataset: {e}")

        # Load tool dataset
        if args.nemotron_tool_fraction > 0:
            try:
                if Path(args.local_tool_path).exists():
                    tool_ds = load_from_disk(args.local_tool_path).to_iterable_dataset()
                else:
                    tool_ds = load_dataset("nvidia/HelpSteer2", split="train", streaming=True)
                datasets_to_load.append(tool_ds)
                probabilities.append(args.nemotron_tool_fraction)
                print(f"  Loaded nemotron-tool (fraction: {args.nemotron_tool_fraction})")
            except Exception as e:
                print(f"  WARNING: Could not load tool dataset: {e}")

        if not datasets_to_load:
            raise ValueError("No datasets loaded for nemotron-mixed")

        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        # Interleave datasets
        dataset = interleave_datasets(datasets_to_load, probabilities=probabilities, stopping_strategy="all_exhausted")
        dataset_type = "nemotron"

    elif dataset_name.startswith("nemotron-"):
        # Single nemotron dataset
        subset = dataset_name.replace("nemotron-", "")
        local_path_map = {
            "code": args.local_code_path,
            "math": args.local_math_path,
            "tool": args.local_tool_path,
        }

        if subset in local_path_map and Path(local_path_map[subset]).exists():
            print(f"Loading {dataset_name} from local path: {local_path_map[subset]}")
            dataset = load_from_disk(local_path_map[subset]).to_iterable_dataset()
        else:
            print(f"Loading {dataset_name} from HuggingFace (fallback)")
            dataset = load_dataset("nvidia/HelpSteer2", split="train", streaming=True)

        dataset_type = "nemotron"

    elif dataset_name == "c4":
        print("Loading C4 validation split")
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        dataset_type = "text"

    else:
        print(f"Loading {dataset_name} as generic dataset")
        dataset = load_dataset(dataset_name, split="validation", streaming=True)
        dataset_type = "text"

    return dataset, dataset_type


def extract_text_from_sample(sample: Dict, dataset_type: str) -> str:
    """
    Extract text from sample based on dataset type.

    Args:
        sample: Dataset sample dict
        dataset_type: 'text' or 'nemotron'

    Returns:
        Extracted text string
    """
    if dataset_type == 'nemotron':
        # Nemotron format 1: prompt/response (HelpSteer2 format)
        if 'prompt' in sample and 'response' in sample:
            prompt = sample['prompt'].strip()
            response = sample['response'].strip()
            if prompt and response:
                return f"<instruction> {prompt}\n<response> {response}"
            elif response:
                return response
            elif prompt:
                return prompt

        # Nemotron format 2: messages with instruction/response
        elif 'messages' in sample and sample['messages']:
            text_parts = []
            for turn in sample['messages']:
                role = turn.get('role', '')
                content = turn.get('content', '')
                if role == 'user':
                    text_parts.append(f"<instruction> {content}")
                elif role == 'assistant':
                    text_parts.append(f"<response> {content}")
            return "\n".join(text_parts)

        # Nemotron format 3: conversations (old format)
        elif 'conversations' in sample and sample['conversations']:
            text_parts = []
            for turn in sample['conversations']:
                role = turn.get('role', turn.get('from', ''))
                content = turn.get('content', turn.get('value', ''))
                if role in ['user', 'human']:
                    text_parts.append(f"<instruction> {content}")
                elif role in ['assistant', 'gpt']:
                    text_parts.append(f"<response> {content}")
            return "\n".join(text_parts)

        else:
            # Fallback to plain text field
            return sample.get('text', '')

    else:
        # Plain text format (C4, etc.)
        return sample.get('text', '')


def calculate_token_distribution(token_ids: List[int]) -> Counter:
    """Calculate token frequency distribution."""
    return Counter(token_ids)


def distribution_distance(dist1: Counter, dist2: Counter) -> float:
    """
    Calculate KL divergence between two distributions.
    Returns normalized distance score.
    """
    all_tokens = set(dist1.keys()) | set(dist2.keys())
    total1 = sum(dist1.values())
    total2 = sum(dist2.values())

    if total1 == 0 or total2 == 0:
        return float('inf')

    kl_div = 0.0
    epsilon = 1e-10

    for token in all_tokens:
        p = (dist1.get(token, 0) + epsilon) / (total1 + epsilon * len(all_tokens))
        q = (dist2.get(token, 0) + epsilon) / (total2 + epsilon * len(all_tokens))
        kl_div += p * np.log(p / q)

    return kl_div


def score_candidate_batch(args):
    """
    Score a batch of candidates for parallel processing.

    Args:
        args: Tuple of (candidate_indices, candidates, cumulative_dist, global_dist)

    Returns:
        List of (index, score) tuples
    """
    candidate_indices, candidates, cumulative_dist, global_dist = args
    results = []

    for idx in candidate_indices:
        candidate = candidates[idx]

        # Simulate adding this sample
        test_dist = cumulative_dist.copy()
        test_dist.update(candidate["token_dist"])

        # Score: minimize distance to global distribution
        dist_score = -distribution_distance(test_dist, global_dist)

        # Bonus for longer samples (more efficient)
        length_bonus = 0.01 * candidate["num_tokens"]

        score = dist_score + length_bonus
        results.append((idx, score))

    return results


def select_diverse_samples(
    tokenizer,
    dataset,
    target_tokens: int,
    max_candidates: int = 10000,
    max_seq_length: int = 512,
    dataset_type: str = 'text',
    num_workers: int = None,
    batch_select: int = 1,
) -> List[Dict]:
    """
    Select samples that provide good token distribution coverage.

    Strategy:
    1. Tokenize a pool of candidates
    2. Calculate token distribution for each
    3. Greedily select samples that maximize distribution diversity

    Args:
        tokenizer: Tokenizer to use
        dataset: Streaming dataset
        target_tokens: Target number of tokens to collect
        max_candidates: Maximum candidates to consider
        max_seq_length: Maximum sequence length
        dataset_type: Type of dataset ('text' or 'nemotron')
        num_workers: Number of parallel workers (default: CPU count)
        batch_select: Number of samples to select per iteration (higher = faster, less optimal)

    Returns:
        List of selected samples with tokenized inputs
    """
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Collecting {max_candidates} candidate samples...")
    print(f"Using {num_workers} workers for parallel scoring")
    if batch_select > 1:
        print(f"Batch selection: selecting top-{batch_select} per iteration for speed")

    # Phase 1: Collect and tokenize candidates
    candidates = []
    samples_seen = 0
    samples_too_short = 0
    samples_empty = 0

    for sample in tqdm(dataset, total=max_candidates, desc="Tokenizing candidates"):
        samples_seen += 1

        if len(candidates) >= max_candidates:
            break

        text = extract_text_from_sample(sample, dataset_type)

        if len(text) == 0:
            samples_empty += 1
            continue

        if len(text) < 50:  # Skip very short texts
            samples_too_short += 1
            continue

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )

        input_ids = inputs["input_ids"][0].tolist()

        # Skip very short tokenized samples
        if len(input_ids) < 10:
            continue

        # Calculate token distribution for this sample
        token_dist = calculate_token_distribution(input_ids)

        candidates.append({
            "text": text,
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0].tolist(),
            "num_tokens": len(input_ids),
            "token_dist": token_dist,
        })

    print(f"\nCandidate collection summary:")
    print(f"  Samples seen: {samples_seen}")
    print(f"  Samples with empty text: {samples_empty}")
    print(f"  Samples too short (<50 chars): {samples_too_short}")
    print(f"  Valid candidates collected: {len(candidates)}")

    if len(candidates) == 0:
        raise ValueError(
            f"No valid candidates found after processing {samples_seen} samples. "
            f"All samples had empty or too-short text. Check dataset structure and text extraction logic."
        )

    # Phase 2: Greedy selection for diversity with parallel scoring
    print(f"\nSelecting diverse subset to reach {target_tokens} tokens...")

    selected = []
    selected_tokens = 0
    cumulative_dist = Counter()

    # Calculate global distribution across all candidates
    global_dist = Counter()
    for candidate in candidates:
        global_dist.update(candidate["token_dist"])

    # Create process pool for parallel scoring
    with Pool(processes=num_workers) as pool:
        with tqdm(total=target_tokens, desc="Selecting samples") as pbar:
            while selected_tokens < target_tokens and len(candidates) > 0:
                # Split candidates into batches for parallel processing
                num_candidates = len(candidates)
                batch_size = max(1, num_candidates // num_workers)
                batches = []

                for i in range(0, num_candidates, batch_size):
                    batch_indices = list(range(i, min(i + batch_size, num_candidates)))
                    batches.append((batch_indices, candidates, cumulative_dist, global_dist))

                # Score all candidates in parallel
                results = pool.map(score_candidate_batch, batches)

                # Flatten results and find best
                all_scores = []
                for batch_results in results:
                    all_scores.extend(batch_results)

                # Sort by score and select top-N (batch_select)
                all_scores.sort(key=lambda x: x[1], reverse=True)
                to_select = min(batch_select, len(all_scores), len(candidates))

                # Select top candidates
                selected_indices = [all_scores[i][0] for i in range(to_select)]
                # Sort indices in reverse to pop from end (preserves earlier indices)
                selected_indices.sort(reverse=True)

                for idx in selected_indices:
                    selected_sample = candidates.pop(idx)
                    selected.append(selected_sample)
                    cumulative_dist.update(selected_sample["token_dist"])
                    selected_tokens += selected_sample["num_tokens"]
                    pbar.update(selected_sample["num_tokens"])

                    # Break if we've reached target
                    if selected_tokens >= target_tokens:
                        break

    print(f"\nSelected {len(selected)} samples totaling {selected_tokens} tokens")

    # Calculate final distribution statistics
    vocab_size = len(cumulative_dist)
    coverage = vocab_size / len(global_dist) if len(global_dist) > 0 else 0
    print(f"Vocabulary coverage: {vocab_size}/{len(global_dist)} tokens ({coverage:.1%})")

    return selected


def collect_router_probabilities(
    model,
    samples: List[Dict],
    device: str = "cuda",
) -> Tuple[np.ndarray, Dict]:
    """
    Collect router probabilities for all layers across samples.

    Args:
        model: gpt-oss model with output_router_logits=True
        samples: List of pre-tokenized samples
        device: Device to use

    Returns:
        Tuple of:
        - router_probs: [num_tokens, num_layers, num_experts] array
        - metadata: Dictionary with collection statistics
    """
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts

    # Pre-allocate arrays
    total_tokens = sum(s["num_tokens"] for s in samples)
    router_probs = np.zeros((total_tokens, num_layers, num_experts), dtype=np.float16)

    print(f"\nCollecting router probabilities for {total_tokens} tokens...")
    print(f"Model: {num_layers} layers, {num_experts} experts per layer")

    model.eval()
    token_idx = 0

    with torch.no_grad():
        for sample in tqdm(samples, desc="Processing samples"):
            # Prepare inputs
            inputs = {
                "input_ids": torch.tensor([sample["input_ids"]], device=device),
                "attention_mask": torch.tensor([sample["attention_mask"]], device=device),
            }

            # Forward pass with router logits
            outputs = model(**inputs, output_router_logits=True)

            # Extract router probabilities for each layer
            if outputs.router_logits is not None:
                # Check shape of first layer's router logits
                first_logits = outputs.router_logits[0]

                if first_logits.dim() == 3:
                    # Shape: (batch_size, seq_len, num_experts)
                    batch_size, seq_len, _ = first_logits.shape
                elif first_logits.dim() == 2:
                    # Shape: (seq_len, num_experts) - no batch dimension
                    seq_len, _ = first_logits.shape
                    batch_size = 1
                else:
                    raise ValueError(f"Unexpected router_logits shape: {first_logits.shape}")

                for layer_idx, router_logits in enumerate(outputs.router_logits):
                    # Convert to probabilities
                    probs = F.softmax(router_logits, dim=-1)

                    # Extract probabilities (handle both 2D and 3D)
                    if probs.dim() == 3:
                        probs_np = probs[0].cpu().float().numpy().astype(np.float16)
                    else:
                        probs_np = probs.cpu().float().numpy().astype(np.float16)

                    router_probs[token_idx:token_idx + seq_len, layer_idx, :] = probs_np

            token_idx += seq_len

    # Collect metadata
    metadata = {
        "num_samples": len(samples),
        "num_tokens": total_tokens,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": model.config.num_experts_per_tok,
        "model_name": model.config._name_or_path if hasattr(model.config, '_name_or_path') else "unknown",
    }

    return router_probs, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Collect router probabilities with distribution-aware sampling"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/mnt/models/gpt-oss-120b",
        help="Path to gpt-oss model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nemotron-mixed",
        help="Dataset to use: c4, nemotron-code, nemotron-math, nemotron-tool, or nemotron-mixed (default: nemotron-mixed)",
    )
    parser.add_argument(
        "--nemotron-code-fraction",
        type=float,
        default=0.4,
        help="Fraction of samples from nemotron-code when using nemotron-mixed (default: 0.4)",
    )
    parser.add_argument(
        "--nemotron-math-fraction",
        type=float,
        default=0.4,
        help="Fraction of samples from nemotron-math when using nemotron-mixed (default: 0.4)",
    )
    parser.add_argument(
        "--nemotron-tool-fraction",
        type=float,
        default=0.2,
        help="Fraction of samples from nemotron-tool when using nemotron-mixed (default: 0.2)",
    )
    parser.add_argument(
        "--local-code-path",
        type=str,
        default="/mnt/git/gpt-oss-shared/data/nvidia_synthetic_code_instructions",
        help="Local path to nemotron-code dataset",
    )
    parser.add_argument(
        "--local-math-path",
        type=str,
        default="/mnt/git/gpt-oss-shared/data/nvidia_synthetic_math_instructions",
        help="Local path to nemotron-math dataset",
    )
    parser.add_argument(
        "--local-tool-path",
        type=str,
        default="/mnt/git/gpt-oss-shared/data/nvidia_synthetic_tool_calling_v1_instructions",
        help="Local path to nemotron-tool dataset",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=100000,
        help="Target number of tokens to collect",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10000,
        help="Number of candidate samples to consider for selection",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/router_probs.npz",
        help="Output file for router probabilities (NPZ format)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers for sample selection (default: CPU count)",
    )
    parser.add_argument(
        "--batch-select",
        type=int,
        default=1,
        help="Number of samples to select per iteration (higher = faster but less optimal, default: 1)",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Router Probability Collection with Distribution-Aware Sampling")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Target tokens: {args.target_tokens:,}")
    print(f"Max candidates: {args.max_candidates:,}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print("=" * 80)

    # Load tokenizer (needed for sampling phase)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset, dataset_type = load_dataset_by_name(args.dataset, args)

    # Phase 1: Distribution-aware sample selection
    selected_samples = select_diverse_samples(
        tokenizer=tokenizer,
        dataset=dataset,
        target_tokens=args.target_tokens,
        max_candidates=args.max_candidates,
        max_seq_length=args.max_seq_length,
        dataset_type=dataset_type,
        num_workers=args.num_workers,
        batch_select=args.batch_select,
    )

    # Phase 2: Load model and collect probabilities
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        output_router_logits=True,
    )

    print(f"Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_local_experts} experts per layer")

    # Collect router probabilities
    router_probs, metadata = collect_router_probabilities(
        model=model,
        samples=selected_samples,
        device=args.device,
    )

    # Phase 3: Save results
    print(f"\nSaving results to {args.output}")
    np.savez_compressed(
        args.output,
        router_probs=router_probs,
        metadata=json.dumps(metadata),
    )

    # Also save metadata as JSON for easy inspection
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("Collection Summary")
    print("=" * 80)
    print(f"Total tokens collected: {metadata['num_tokens']:,}")
    print(f"Number of samples: {metadata['num_samples']}")
    print(f"Router probabilities shape: {router_probs.shape}")
    print(f"Storage size: {router_probs.nbytes / (1024**2):.1f} MB (compressed)")
    print("\nCollection complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
