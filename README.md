# NEXUS: Neural Expert Unified Specialization

**Efficient fine-tuning for Mixture-of-Experts models on minimal hardware**

NEXUS enables domain specialization of massive MoE models (100B+ parameters) by training a small shared expert (~1B params) while keeping routed experts frozen. Like LoRA for mixture-of-experts, but architecturally integrated.

---

## Key Features

✅ **Minimal Hardware**: Train 120B MoE on 2×98GB consumer GPUs
✅ **Intelligent Initialization**: PCA-guided expert selection and averaging
✅ **Flexible**: Works with frozen or trainable routers
✅ **Extensible**: Model-family architecture (GPT-OSS, Llama-MoE, Qwen-MoE, ...)
✅ **Production Ready**: Offline distillation, DDP, advanced scheduling

---

## Quick Start

### Installation

**For GPT-OSS with MXFP4** (requires forked transformers):

```bash
# 1. Install forked transformers with MXFP4 + shared expert support
pip install git+https://github.com/eous/transformers.git@gpt-oss-mxfp4

# 2. Install NEXUS
git clone https://github.com/eous/nexus.git
cd nexus
pip install -e .
```

**Note**: The fork adds ~100 lines for:
- `GptOssSharedExpert` support in MoE layers
- `Mxfp4Config(dequantize=True)` for weight extraction during conversion

Transformers modifications are documented in the fork's branch.

### Basic Workflow (GPT-OSS)

```bash
# 1. Collect router probabilities (1M tokens, ~4 hours)
python scripts/gpt_oss/collect_router_probs.py \
    --model /path/to/gpt-oss-120b \
    --target-tokens 1000000 \
    --output data/router_probs.npz

# 2. PCA analysis - select top-24 diverse experts (~5 min)
python scripts/gpt_oss/analyze_pca.py \
    --input data/router_probs.npz \
    --output data/pca_stats.json \
    --top-k 24

# 3. Convert model - add shared expert (~15 min)
python scripts/gpt_oss/convert.py \
    --input /path/to/gpt-oss-120b \
    --output /path/to/gpt-oss-120b-nexus \
    --pca-stats data/pca_stats.json

# 4. Train shared expert (~hours)
python scripts/gpt_oss/train.py \
    --student-model /path/to/gpt-oss-120b-nexus \
    --teacher-model /path/to/gpt-oss-120b \
    --freeze-router \
    --output-dir outputs/specialized
```

---

## What Makes NEXUS Different?

| Method | Approach | Trainable | Use Case |
|--------|----------|-----------|----------|
| **LoRA** | Low-rank updates to existing weights | ~100M | General adapter for any model |
| **NEXUS** | Add always-active shared expert to MoE | ~900M | MoE-specific, captures common patterns |
| **Full FT** | Update all parameters | 120B | Datacenter-scale only |

**NEXUS is to MoE what LoRA is to dense models**: efficient parameter addition with architectural integration.

---

## Architecture

### Standard MoE (4-of-128 routing)
```
Per token: Router selects 4 experts from 128
Active params: ~3.75B (4/128 × 120B)
```

### NEXUS Enhanced MoE (4+1 architecture)
```
Per token:
├─ Router selects 4 experts from 128 (frozen MXFP4)
└─ Shared expert ALWAYS active (trainable BF16)

Active params: ~3.75B (routed) + ~25M (shared) ≈ 3.8B
Trainable: ~900M (shared expert only with --freeze-router)
```

**The shared expert** learns common patterns across all tokens, allowing routed experts to specialize further.

---

## Supported Models

### Currently Supported
- ✅ **GPT-OSS 120B** (36 layers, 4-of-128 routing)
- ✅ **GPT-OSS 20B** (24 layers, 4-of-128 routing)

Model variant is automatically detected from config.

### Planned Support
- 🔄 **Llama-MoE** (coming soon)
- 🔄 **Qwen-MoE** (coming soon)
- 🔄 **DeepSeek-V3** (coming soon)

Adding new model families requires implementing family-specific conversion logic in `nexus/models/<family>/`.

---

## Key Innovations

### 1. PCA-Guided Expert Selection
Instead of random initialization or simple averaging, NEXUS uses PCA on router activation patterns to identify the most diverse and important experts for merging.

**Discovery**: On GPT-OSS 120B, experts are **highly specialized**:
- Top-4 experts: 11.5% importance
- Top-24 experts: 43% importance
- **Conclusion**: Need 24 experts to capture meaningful patterns

### 2. Distribution-Aware Sampling
When collecting router statistics, NEXUS tokenizes 10K candidates and greedily selects a subset that maximizes vocabulary coverage (10× speedup with parallelization).

### 3. Frozen Router Training
For pre-trained MoE models, freezing the router while training the shared expert:
- ✅ Simpler optimization
- ✅ No routing destabilization
- ✅ Shared expert learns to "fill gaps" in existing routing

---

## Hardware Requirements

### Conversion (One-time)
- **CPU**: 250GB RAM (for PCA weight computation)
- **GPU**: 75GB VRAM (for final model assembly)
- **Time**: ~4-5 hours total

### Training
- **GPUs**: 2×98GB VRAM (DDP across consumer GPUs)
- **Memory**: ~100-110GB total for 120B model
- **Time**: Depends on steps (1000 steps ≈ 2-3 hours)

---

## Citation

If you use NEXUS in your research, please cite:

```bibtex
@software{nexus2025,
  title={NEXUS: Neural Expert Unified Specialization},
  author={Patrick},
  year={2025},
  url={https://github.com/eous/nexus}
}
```

---

## Documentation

- [Architecture Deep Dive](docs/architecture.md) - Design decisions and technical details
- [GPT-OSS Quick Reference](docs/models/gpt_oss.md) - Model-specific documentation
- [Quickstart Guide](docs/quickstart.md) - Step-by-step tutorial
- [CLAUDE.md](CLAUDE.md) - Development guide for AI assistants

---

## Examples

See [`examples/gpt_oss_workflow.py`](examples/gpt_oss_workflow.py) for a complete GPT-OSS training workflow example.

---

## Contributing

NEXUS is designed to be extensible to new MoE architectures. To add support for a new model family:

1. Create `nexus/models/<family>/` directory
2. Implement family-specific conversion logic
3. Add training adaptations if needed
4. Update documentation

See [Contributing Guide](CONTRIBUTING.md) for details.

---

## License

MIT License - See LICENSE file

---

## Acknowledgments

- DeepSeek-V3 for shared expert architecture inspiration
- HuggingFace Transformers for model infrastructure
- GPT-OSS team for the base model architecture
