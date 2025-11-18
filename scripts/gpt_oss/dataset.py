#!/usr/bin/env python3
"""
Dataset utilities for GPT-OSS training.

Supports:
- Multiple HuggingFace datasets (Nemotron, C4)
- Efficient streaming with tokenization
- Local dataset paths (avoids downloads)
- Worker-based parallelism
- Tokenization caching
- Proper attention masks

Based on DeepSeek-V3 dataset patterns adapted for HuggingFace datasets.
"""

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import torch
from datasets import interleave_datasets, load_dataset, load_from_disk
from torch.utils.data import DataLoader, IterableDataset


class StreamingTextDataset(IterableDataset):
    """
    Memory-efficient streaming dataset that tokenizes on-the-fly.

    Features:
    - Streams from HuggingFace datasets (no memory preload)
    - Tokenization caching with LRU eviction
    - Handles plain text (C4) and conversation formats (Nemotron)
    - Multi-worker support
    - Proper padding and attention masks
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int = 512,
        max_samples: Optional[int] = None,
        dataset_type: str = 'text',
        cache_size: int = 1000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize streaming dataset.

        Args:
            dataset: HuggingFace streaming dataset or iterable
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            max_samples: Maximum number of samples to yield (None = unlimited)
            dataset_type: 'text' for plain text, 'nemotron' for conversation format
            cache_size: Size of tokenization cache (default: 1000)
            logger: Logger instance for debugging
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.dataset_type = dataset_type
        self.cache_size = cache_size
        self.logger = logger or logging.getLogger(__name__)

        # Tokenization cache (shared across iterations in same process)
        self._token_cache = OrderedDict()
        self._cache_stats = {'hits': 0, 'misses': 0}

    def _extract_text(self, sample: Dict) -> str:
        """
        Extract text from sample based on dataset type.

        Args:
            sample: Dataset sample dict

        Returns:
            Extracted text string
        """
        if self.dataset_type == 'nemotron':
            # Nemotron format: messages with role/content pairs
            # Uses simple role-based formatting that preserves conversation structure
            if 'messages' in sample and sample['messages']:
                # FIRST: Validate entire conversation
                # If any user/assistant message is empty/invalid, skip ENTIRE sample
                # This prevents malformed conversations (e.g., consecutive assistant messages)
                for msg in sample['messages']:
                    role = msg.get('role', '')
                    content = msg.get('content', '')

                    if role in ['user', 'assistant']:
                        if not content or content == '-':
                            # Invalid user/assistant message - skip entire sample
                            return ''  # Empty string signals invalid sample

                # All user/assistant messages valid - extract text
                text_parts = []
                for msg in sample['messages']:
                    role = msg.get('role', '')
                    content = msg.get('content', '')

                    # Format: role\ncontent\n (simple, preserves structure)
                    if content:
                        text_parts.append(f"{role}\n{content}")
                    else:
                        # Empty system/tool message - keep role marker only
                        text_parts.append(f"{role}\n")

                return "\n".join(text_parts)

            elif 'conversations' in sample and sample['conversations']:
                # Fallback for old conversation format
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

    def _tokenize_cached(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize text with caching.

        Args:
            text: Text to tokenize

        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Create cache key from text hash
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

        # Check cache
        if text_hash in self._token_cache:
            self._cache_stats['hits'] += 1
            # Move to end (LRU)
            self._token_cache.move_to_end(text_hash)
            return self._token_cache[text_hash].copy()

        # Cache miss - tokenize
        self._cache_stats['misses'] += 1

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Create labels (mask padding tokens with -100 to ignore in loss)
        labels = encoded['input_ids'].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': labels
        }

        # Add to cache
        self._token_cache[text_hash] = result.copy()

        # Maintain cache size (LRU eviction)
        if len(self._token_cache) > self.cache_size:
            self._token_cache.popitem(last=False)  # Remove oldest

        return result

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through dataset, yielding tokenized samples."""
        samples_yielded = 0

        for sample in self.dataset:
            if self.max_samples is not None and samples_yielded >= self.max_samples:
                break

            # Extract text based on dataset type
            text = self._extract_text(sample)

            # Skip sample if text extraction failed or returned empty
            # (e.g., invalid messages, empty user/assistant content)
            if not text:
                continue

            # Tokenize with caching
            try:
                batch = self._tokenize_cached(text)
                yield batch
                samples_yielded += 1

            except Exception as e:
                self.logger.warning(f"Tokenization failed for sample {samples_yielded}: {e}")
                continue

        # Log final cache statistics
        if self._cache_stats['hits'] + self._cache_stats['misses'] > 0:
            hit_rate = self._cache_stats['hits'] / (self._cache_stats['hits'] + self._cache_stats['misses'])
            self.logger.debug(f"Tokenization cache hit rate: {hit_rate:.2%}")

    def get_cache_stats(self) -> Dict:
        """Get tokenization cache statistics."""
        return self._cache_stats.copy()


def create_nemotron_dataset(
    tokenizer,
    splits: List[str] = ['code', 'math', 'tool_calling'],
    local_paths: Optional[Dict[str, str]] = None,
    streaming: bool = True,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Create interleaved Nemotron dataset (code + math + tool_calling).

    Uses nvidia/Nemotron-Post-Training-Dataset-v2 for code/math (newer data).
    Uses nvidia/Nemotron-Post-Training-Dataset-v1 for tool_calling (not in v2).

    Args:
        tokenizer: Tokenizer to use
        splits: List of splits to use (e.g., ['code', 'math', 'tool_calling'])
        local_paths: Dict with local paths {'code': path, 'math': path, 'tool': path}
        streaming: Whether to use streaming mode
        max_length: Maximum sequence length
        max_samples: Maximum samples per dataset
        logger: Logger instance

    Returns:
        StreamingTextDataset wrapping interleaved HF datasets
    """
    logger = logger or logging.getLogger(__name__)

    datasets_to_load = []

    # Map split names to dataset loading logic
    split_map = {
        'code': ('code', 'code'),
        'math': ('math', 'math'),
        'tool_calling': ('tool', 'tool_calling'),
        'tool': ('tool', 'tool_calling'),  # Alias
    }

    for split in splits:
        if split not in split_map:
            logger.warning(f"Unknown Nemotron split: {split}, skipping")
            continue

        local_key, hf_split = split_map[split]

        # Check for local path first
        if local_paths and local_key in local_paths and local_paths[local_key]:
            logger.info(f"  Loading {split} from local: {local_paths[local_key]}")
            ds = load_from_disk(local_paths[local_key])
            if not streaming:
                # Keep as-is
                pass
            else:
                # Convert to iterable
                ds = ds.to_iterable_dataset()
        else:
            # Load from HuggingFace
            # Use v2 for code/math (newer data), v1 for tool_calling (not in v2)
            if hf_split == 'tool_calling':
                logger.info(f"  Streaming {split} from HuggingFace (v1 - tool_calling not in v2)")
                ds = load_dataset(
                    "nvidia/Nemotron-Post-Training-Dataset-v1",
                    split=hf_split,
                    streaming=streaming
                )
            else:
                logger.info(f"  Streaming {split} from HuggingFace (v2)")
                ds = load_dataset(
                    "nvidia/Nemotron-Post-Training-Dataset-v2",
                    split=hf_split,
                    streaming=streaming
                )

        datasets_to_load.append(ds)

    # Interleave datasets for diversity
    if len(datasets_to_load) > 1:
        hf_dataset = interleave_datasets(datasets_to_load)
        logger.info(f"  ✓ Interleaved {len(datasets_to_load)} Nemotron datasets")
    else:
        hf_dataset = datasets_to_load[0]
        logger.info(f"  ✓ Using single Nemotron dataset")

    # Wrap in streaming dataset
    return StreamingTextDataset(
        dataset=hf_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        dataset_type='nemotron',
        logger=logger
    )


def create_c4_dataset(
    tokenizer,
    split: str = 'train',
    streaming: bool = True,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Create C4 English text dataset.

    Args:
        tokenizer: Tokenizer to use
        split: Dataset split ('train', 'validation')
        streaming: Whether to use streaming mode
        max_length: Maximum sequence length
        max_samples: Maximum samples
        logger: Logger instance

    Returns:
        StreamingTextDataset wrapping C4
    """
    logger = logger or logging.getLogger(__name__)

    logger.info(f"  Streaming C4 (en) split: {split}")
    hf_dataset = load_dataset("allenai/c4", "en", split=split, streaming=streaming)

    return StreamingTextDataset(
        dataset=hf_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        dataset_type='text',
        logger=logger
    )


def create_dataloader(
    tokenizer,
    batch_size: int = 1,
    max_samples: Optional[int] = 100000,
    seq_len: int = 512,
    num_workers: int = 0,
    dataset_name: str = 'nemotron',
    local_dataset_paths: Optional[Dict[str, str]] = None,
    nemotron_splits: Optional[List[str]] = None,
    c4_split: str = 'train',
    logger: Optional[logging.Logger] = None
) -> DataLoader:
    """
    Create memory-efficient streaming DataLoader for GPT-OSS training.

    Args:
        tokenizer: Tokenizer to use
        batch_size: Batch size
        max_samples: Maximum samples (None = unlimited, controlled by training steps)
        seq_len: Maximum sequence length
        num_workers: Number of DataLoader workers (0 = main process only)
        dataset_name: 'nemotron' or 'c4'
        local_dataset_paths: Dict with local paths for Nemotron datasets
                            {'code': path, 'math': path, 'tool': path}
        nemotron_splits: List of Nemotron splits to use (default: ['code', 'math', 'tool_calling'])
        c4_split: C4 split to use (default: 'train')
        logger: Logger instance

    Returns:
        DataLoader with streaming dataset

    Example:
        # Default nemotron (code + math + tool_calling)
        loader = create_dataloader(tokenizer, batch_size=4, seq_len=1024)

        # Custom nemotron splits
        loader = create_dataloader(
            tokenizer,
            dataset_name='nemotron',
            nemotron_splits=['code', 'math'],  # Only code + math
            batch_size=4,
            seq_len=512
        )

        # C4 validation split
        loader = create_dataloader(
            tokenizer,
            dataset_name='c4',
            c4_split='validation',
            batch_size=8
        )

        # Local nemotron datasets (no download)
        loader = create_dataloader(
            tokenizer,
            dataset_name='nemotron',
            local_dataset_paths={
                'code': '/mnt/data/nemotron/code',
                'math': '/mnt/data/nemotron/math',
                'tool': '/mnt/data/nemotron/tool'
            }
        )
    """
    logger = logger or logging.getLogger(__name__)

    logger.info(f"Creating {dataset_name} dataset...")
    logger.info(f"  Max samples: {max_samples:,}" if max_samples else "  Max samples: unlimited")
    logger.info(f"  Sequence length: {seq_len}")
    logger.info(f"  Batch size: {batch_size}")

    # Create dataset based on type
    if dataset_name == 'nemotron':
        # Default splits if not specified
        if nemotron_splits is None:
            nemotron_splits = ['code', 'math', 'tool_calling']

        logger.info(f"  Nemotron splits: {', '.join(nemotron_splits)}")

        stream_dataset = create_nemotron_dataset(
            tokenizer=tokenizer,
            splits=nemotron_splits,
            local_paths=local_dataset_paths,
            streaming=True,
            max_length=seq_len,
            max_samples=max_samples,
            logger=logger
        )

    elif dataset_name == 'c4':
        logger.info(f"  C4 split: {c4_split}")

        stream_dataset = create_c4_dataset(
            tokenizer=tokenizer,
            split=c4_split,
            streaming=True,
            max_length=seq_len,
            max_samples=max_samples,
            logger=logger
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'nemotron' or 'c4'")

    # Create DataLoader
    dataloader = DataLoader(
        stream_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    logger.info(f"✓ DataLoader created (streaming mode - no memory preload)")

    return dataloader


def create_custom_dataset(
    tokenizer,
    datasets: List[str],
    splits: List[str],
    local_paths: Optional[Dict[str, str]] = None,
    max_length: int = 512,
    max_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Create custom interleaved dataset from multiple sources.

    This provides fine-grained control over dataset composition.

    Args:
        tokenizer: Tokenizer to use
        datasets: List of dataset names (e.g., ['nemotron-code', 'c4', 'nemotron-math'])
        splits: List of splits for each dataset (e.g., ['train', 'validation', 'train'])
        local_paths: Dict with local paths for Nemotron datasets
        max_length: Maximum sequence length
        max_samples: Maximum samples total
        logger: Logger instance

    Returns:
        StreamingTextDataset wrapping interleaved datasets

    Example:
        # Mix code, math, and C4 validation
        dataset = create_custom_dataset(
            tokenizer,
            datasets=['nemotron-code', 'nemotron-math', 'c4'],
            splits=['train', 'train', 'validation'],
            max_length=1024
        )
    """
    logger = logger or logging.getLogger(__name__)

    if len(datasets) != len(splits):
        raise ValueError(f"datasets ({len(datasets)}) and splits ({len(splits)}) must have same length")

    logger.info(f"Creating custom interleaved dataset from {len(datasets)} sources")

    datasets_to_load = []

    for dataset_name, split in zip(datasets, splits):
        logger.info(f"  Loading {dataset_name} (split: {split})")

        if dataset_name == 'nemotron-code':
            if local_paths and 'code' in local_paths and local_paths['code']:
                ds = load_from_disk(local_paths['code'])
                ds = ds.to_iterable_dataset()
            else:
                ds = load_dataset(
                    "nvidia/Nemotron-Post-Training-Dataset-v2",
                    split='code',
                    streaming=True
                )

        elif dataset_name == 'nemotron-math':
            if local_paths and 'math' in local_paths and local_paths['math']:
                ds = load_from_disk(local_paths['math'])
                ds = ds.to_iterable_dataset()
            else:
                ds = load_dataset(
                    "nvidia/Nemotron-Post-Training-Dataset-v2",
                    split='math',
                    streaming=True
                )

        elif dataset_name == 'nemotron-tool':
            if local_paths and 'tool' in local_paths and local_paths['tool']:
                ds = load_from_disk(local_paths['tool'])
                ds = ds.to_iterable_dataset()
            else:
                # Note: tool_calling not available in v2, using v1
                ds = load_dataset(
                    "nvidia/Nemotron-Post-Training-Dataset-v1",
                    split='tool_calling',
                    streaming=True
                )

        elif dataset_name == 'c4':
            ds = load_dataset("allenai/c4", "en", split=split, streaming=True)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        datasets_to_load.append(ds)

    # Interleave datasets
    if len(datasets_to_load) > 1:
        hf_dataset = interleave_datasets(datasets_to_load)
        logger.info(f"  ✓ Interleaved {len(datasets_to_load)} datasets")
    else:
        hf_dataset = datasets_to_load[0]
        logger.info(f"  ✓ Using single dataset")

    # Wrap in streaming dataset (detect type from first dataset name)
    dataset_type = 'nemotron' if any('nemotron' in d for d in datasets) else 'text'

    return StreamingTextDataset(
        dataset=hf_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
        dataset_type=dataset_type,
        logger=logger
    )
