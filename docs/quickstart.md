# NEXUS Quickstart Guide

This guide walks you through specializing a 120B GPT-OSS model on consumer hardware.

## Prerequisites

- **Hardware**: 2×98GB GPUs (e.g., RTX Pro 6000 Blackwell) + 250GB RAM
- **Software**: Python 3.9+, PyTorch 2.0+, CUDA 12+
- **Base model**: Pre-trained GPT-OSS 120B with MXFP4 quantization

## Installation

```bash
git clone https://github.com/yourusername/nexus.git
cd nexus
pip install -e .
```

## Complete Workflow

### Step 1: Collect Router Statistics (4-5 hours)

Analyze how the model routes tokens to understand expert specialization:

```bash
python scripts/gpt_oss/collect_router_probs.py \
    --model /path/to/gpt-oss-120b \
    --target-tokens 1000000 \
    --dataset nemotron-mixed \
    --output data/router_probs.npz
```

**What this does**:
- Collects router probability distributions for 1M tokens
- Uses distribution-aware sampling for vocab coverage
- Parallelized across 96 CPU cores (~650 samples/sec)
- Output: ~500MB compressed NPZ file

### Step 2: PCA Analysis (5-10 minutes)

Identify the most diverse and important experts:

```bash
python scripts/gpt_oss/analyze_pca.py \
    --input data/router_probs.npz \
    --output data/pca_stats.json \
    --top-k 24
```

**What this does**:
- Performs PCA on router probabilities per layer
- Identifies top-24 most important/diverse experts
- Generates diagnostic plots showing:
  - Variance explained by principal components
  - Expert importance distribution (all 128 experts)
  - Cumulative importance curves
- Output: Expert selections + analysis

**Key insight**: If top-4 captures <35% importance → Use top-24 (even distribution)

### Step 3: Convert Model (15-20 minutes)

Add shared expert initialized from PCA-selected experts:

```bash
python scripts/gpt_oss/convert.py \
    --input /path/to/gpt-oss-120b \
    --output /path/to/gpt-oss-120b-nexus \
    --router-stats data/pca_stats.json \
    --init-strategy pca_top24
```

**What this does**:
- **Phase 1** (CPU): Loads model, dequantizes MXFP4, averages 24 experts
- **Phase 2** (GPU): Reloads MXFP4 model, attaches shared expert, saves
- Output: ~59GB model (MXFP4 experts + BF16 shared expert)

**Memory breakdown**:
- Original: 58GB (MXFP4 routed experts)
- With NEXUS: 59GB (+1GB for shared expert)

### Step 4: Train Shared Expert (2-10 hours)

Specialize the shared expert via distillation:

```bash
python scripts/gpt_oss/train.py \
    --student-model /path/to/gpt-oss-120b-nexus \
    --teacher-model /path/to/gpt-oss-120b \
    --freeze-router \
    --use-advanced-scheduler \
    --use-ddp \
    --batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1e-5 \
    --max-steps 10000 \
    --seq-len 1024 \
    --output-dir outputs/specialized
```

**Key flags**:
- `--freeze-router`: Train only shared expert (simpler, recommended)
- `--use-advanced-scheduler`: DeepSeek-V3 style LR/temperature scheduling
- `--use-ddp`: Distribute across 2 GPUs

**Training specs**:
- Trainable: ~896M params (shared expert only)
- Memory: ~100-110GB total across 2 GPUs
- Throughput: ~100-200 tokens/sec (depends on seq_len)

### Step 5: Validate (5-10 minutes)

Test capability retention:

```bash
# Quantitative: perplexity
python scripts/validate_model.py \
    --model outputs/specialized/checkpoint-10000 \
    --baseline /path/to/gpt-oss-120b \
    --compute-perplexity \
    --num-samples 1000

# Qualitative: interactive chat
python scripts/validate_model.py \
    --model outputs/specialized/checkpoint-10000 \
    --chat
```

**Expected results**:
- Perplexity degradation: <5% (excellent retention)
- Router entropy: Similar to baseline
- Qualitative: Test math, code, reasoning tasks

---

## Advanced Usage

### Offline Distillation (2-3× faster)

Pre-compute teacher outputs once, reuse for training:

```bash
# 1. Pre-compute teacher outputs
python scripts/gpt_oss/precompute_teacher.py \
    --model /path/to/gpt-oss-120b \
    --num-steps 10000 \
    --batch-size 1 \
    --seq-len 1024 \
    --output precomputed/teacher_10k

# 2. Train with precomputed outputs (single GPU!)
python scripts/gpt_oss/train.py \
    --student-model /path/to/gpt-oss-120b-nexus \
    --precomputed-teacher precomputed/teacher_10k \
    --freeze-router \
    --batch-size 1 \
    --max-steps 10000 \
    --output-dir outputs/specialized
```

### Trainable Router (DeepSeek-V3 Style)

Let the router co-adapt with the shared expert:

```bash
python scripts/gpt_oss/train.py \
    --student-model /path/to/gpt-oss-120b-nexus \
    --teacher-model /path/to/gpt-oss-120b \
    --use-advanced-scheduler \
    --bias-freeze-ratio 0.5 \
    --output-dir outputs/specialized-adaptive-routing
```

**Differences**:
- Router updates during first 50% of training
- Bias freezing at 50% prevents destabilization
- More complex but allows routing to adapt

---

## Understanding the Results

### PCA Distribution Patterns

**Steep Cliff (>70% in top-4)**:
- Strong expert redundancy
- Top-4 averaging is sufficient

**Moderate (50-70% in top-4)**:
- Balanced distribution
- Use top-4 to top-8

**Gradual Decline (35-50% in top-4)**:
- Some specialization
- Use top-8 to top-16

**Even Distribution (<35% in top-4)** ← GPT-OSS 120B is here:
- High expert specialization (excellent!)
- Use top-24 or more
- Experts are doing their job (not redundant)

### Training Metrics

**Healthy training shows**:
- Total loss: Converging to ~1.5
- KL divergence: 0.06-0.08 (learning, not collapsing)
- Gradient norm: <1.0 most of the time (stable)
- Router L1 distance: 0.2-0.3 (moderate indirect drift)

**Warning signs**:
- KL > 0.15: Potential collapse or divergence
- Gradient spikes >5.0: Instability, reduce LR
- Router L1 > 0.5: Excessive routing changes

---

## Troubleshooting

### OOM during conversion
- **Problem**: GPU runs out of memory loading dequantized model
- **Solution**: Conversion script auto-uses CPU for PCA strategies

### Slow sample selection
- **Problem**: Greedy selection taking hours
- **Solution**: Reduce `--max-candidates` to 2000-5000

### Loss spikes after warmup
- **Problem**: Training unstable during LR transitions
- **Solution**: Increase `--warmup-ratio` to 0.6-0.7

### Router metrics change despite frozen router
- **Normal**: Shared expert changes hidden states → router sees different inputs
- **Expected**: L1 distance ~0.2-0.3 (indirect effect)

---

## Next Steps

- Experiment with different shared expert sizes (1×, 2×, 4× base size)
- Try different top-K selections (16, 24, 32)
- Test on domain-specific datasets (code-only, math-only)
- Compare frozen vs trainable router approaches

For detailed explanations, see:
- [Architecture Guide](architecture.md)
- [GPT-OSS Specifics](models/gpt_oss.md)
- [Training Best Practices](training.md)
