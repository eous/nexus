# Getting Started with NEXUS

Welcome! This guide gets you up and running with NEXUS in 5 minutes.

## Current Status

**NEXUS v0.1.0-alpha** is fully functional for GPT-OSS models:
- ✅ All scripts working (symlinked from `/mnt/git/gpt-oss-shared`)
- ✅ Documentation complete
- ✅ Example workflow provided
- 🔄 Package cleanup needed before PyPI release (see TODO.md)

## Quick Start (Using Your Existing Setup)

Since you already have a trained model, let's validate it:

```bash
cd /mnt/git/nexus

# 1. Validate your trained model
python scripts/gpt_oss/validate.py \
    --model /mnt/git/gpt-oss-shared/outputs/complete_stack/checkpoint-10000 \
    --baseline /mnt/models/gpt-oss-120b \
    --compute-perplexity \
    --num-samples 1000

# 2. Interactive chat to test capabilities
python scripts/gpt_oss/validate.py \
    --model /mnt/git/gpt-oss-shared/outputs/complete_stack/checkpoint-10000 \
    --chat
```

## Full Workflow (From Scratch)

If starting fresh with a new model:

```bash
cd /mnt/git/nexus

# Use the example workflow script
python examples/gpt_oss_workflow.py \
    --model-path /path/to/gpt-oss-120b \
    --output-dir my_specialized_model \
    --target-tokens 100000 \
    --training-steps 1000
```

Or run steps manually:

```bash
# 1. Collect router stats (~30 min for 100K tokens)
python scripts/gpt_oss/collect_router_probs.py \
    --model /path/to/gpt-oss-120b \
    --target-tokens 100000 \
    --output data/router_probs.npz

# 2. PCA analysis (~5 min)
python scripts/gpt_oss/analyze_pca.py \
    --input data/router_probs.npz \
    --output data/pca_stats.json \
    --top-k 24

# 3. Convert model (~15 min)
python scripts/gpt_oss/convert.py \
    --input /path/to/gpt-oss-120b \
    --output models/gpt-oss-120b-nexus \
    --router-stats data/pca_stats.json \
    --init-strategy pca_top24

# 4. Train (~2-3 hours for 1000 steps)
python scripts/gpt_oss/train.py \
    --student-model models/gpt-oss-120b-nexus \
    --teacher-model /path/to/gpt-oss-120b \
    --freeze-router \
    --use-advanced-scheduler \
    --max-steps 1000 \
    --output-dir outputs/trained
```

## Key Files Reference

### Scripts (Symlinked to Working Implementation)
- `scripts/gpt_oss/collect_router_probs.py` - Router statistics collection
- `scripts/gpt_oss/analyze_pca.py` - PCA analysis and expert selection
- `scripts/gpt_oss/convert.py` - Model conversion (add shared expert)
- `scripts/gpt_oss/train.py` - Training with distillation
- `scripts/gpt_oss/validate.py` - Validation and chat

### Documentation
- `README.md` - Project overview
- `docs/quickstart.md` - Detailed workflow guide
- `docs/architecture.md` - Design decisions and comparisons
- `TODO.md` - Development roadmap

### Package
- `nexus/models/gpt_oss/modeling.py` - GptOssSharedExpert implementation
- `setup.py` - Package configuration
- `requirements.txt` - Dependencies

## Configuration Tips

### For Testing (Fast Iteration)
```bash
--target-tokens 10000        # 10K tokens for PCA (~5 min)
--max-candidates 1000        # Smaller candidate pool
--training-steps 100         # Quick training test
```

### For Production (Best Quality)
```bash
--target-tokens 1000000      # 1M tokens for PCA (~4 hours)
--max-candidates 10000       # Full candidate pool
--training-steps 10000       # Full training
```

## Common Workflows

### Experiment 1: Compare Frozen vs Trainable Router
```bash
# Frozen router (simpler)
python scripts/gpt_oss/train.py ... --freeze-router

# Trainable router (DeepSeek-V3 style)
python scripts/gpt_oss/train.py ... --use-advanced-scheduler
```

### Experiment 2: Different Expert Counts
```bash
# Try top-16, top-24, top-32
python scripts/gpt_oss/analyze_pca.py ... --top-k 16
python scripts/gpt_oss/analyze_pca.py ... --top-k 24
python scripts/gpt_oss/analyze_pca.py ... --top-k 32

# Compare perplexities
```

### Experiment 3: Domain Specialization
```bash
# Code-only dataset
python scripts/gpt_oss/collect_router_probs.py \
    --dataset nemotron-code \
    ...

# Math-only dataset
python scripts/gpt_oss/collect_router_probs.py \
    --dataset nemotron-math \
    ...
```

## What's Next?

1. **Review your training results** at step 10K
2. **Test interactively** with `--chat` mode
3. **Compare perplexity** vs baseline
4. **Iterate** on hyperparameters if needed

For questions or issues, see:
- [Architecture Deep Dive](docs/architecture.md)
- [TODO.md](TODO.md) for known limitations
- GitHub issues (when published)

---

**Welcome to NEXUS!** 🚀
