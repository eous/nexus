# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NEXUS** (Neural Expert Unified Specialization) enables efficient fine-tuning of Mixture-of-Experts (MoE) models on minimal hardware by adding a small trainable "shared expert" (~900M params) to massive MoE models (100B+) while keeping routed experts frozen.

**Key Innovation**: Like LoRA for dense models, NEXUS is a parameter-efficient adaptation method specifically designed for MoE architectures.

**Current Status**: v0.1.0-alpha - fully functional for GPT-OSS models (20B and 120B variants) with major refactoring completed (Nov 2025)

## High-Level Architecture

### MoE with Shared Expert (4+1 Architecture)
- **Standard MoE**: Router selects top-4 experts per token from 128 total experts
- **NEXUS Enhanced**: Adds an always-active shared expert that processes ALL tokens
- **Training**: Routed experts frozen (MXFP4 quantized), shared expert trainable (BF16)
- **Memory**: Trains 120B MoE on 2×98GB consumer GPUs

### Training Modes (Recent Refactoring)
The project recently split into two distinct training paths:

**Online Distillation** (`train.py`):
- Teacher on GPU 0, student on GPU 1
- Teacher forward pass computed inline during training
- No DDP - simple gradient accumulation
- Best for: Quick iteration, debugging, <5K steps

**Offline Distillation** (`train_offline.py`):
- Uses precomputed teacher outputs (teacher NOT in memory)
- Multi-GPU DDP training
- Saves ~80GB VRAM
- Best for: Production training, 10K+ steps

### PCA-Guided Initialization (Critical)
- Collect router activation probabilities on diverse data
- Use PCA to identify most important and diverse experts
- Initialize shared expert by averaging top-K experts (typically 24 for GPT-OSS)
- **Never skip this** - random/simple averaging gives poor results

## Core Workflow Pipeline

1. **collect_router_probs.py**: Gather router statistics (~4 hours for 1M tokens)
2. **analyze_pca.py**: PCA analysis and expert selection (~5 min)
3. **convert.py**: Add shared expert to model architecture (~15 min)
4. **train.py OR train_offline.py**: Train shared expert with distillation
5. **validate.py**: Evaluate perplexity and interactive chat

**Supports both GPT-OSS 20B and 120B** - model variant auto-detected from config

## Commands for Development

### Installation
```bash
# Install forked transformers with MXFP4 + shared expert support
pip install git+https://github.com/yourusername/transformers.git@gpt-oss-mxfp4

# Install NEXUS
cd /path/to/nexus
pip install -e .
```

### Basic Training Workflow
```bash
# 1. Collect router statistics
python scripts/gpt_oss/collect_router_probs.py \
    --model /path/to/gpt-oss-120b \
    --target-tokens 1000000 \
    --output data/router_probs.npz

# 2. PCA analysis
python scripts/gpt_oss/analyze_pca.py \
    --input data/router_probs.npz \
    --output data/pca_stats.json \
    --top-k 24

# 3. Convert model
python scripts/gpt_oss/convert.py \
    --input /path/to/gpt-oss-120b \
    --output /path/to/gpt-oss-120b-nexus \
    --pca-stats data/pca_stats.json

# 4a. Online training (simple, 2 GPUs)
python scripts/gpt_oss/train.py \
    --teacher-model /path/to/gpt-oss-120b \
    --student-model /path/to/gpt-oss-120b-nexus \
    --teacher-device cuda:0 \
    --student-device cuda:1 \
    --freeze-router \
    --max-steps 1000 \
    --output-dir outputs/

# 4b. Offline training (faster, DDP, 2+ GPUs)
# First precompute teacher outputs
accelerate launch --num_processes=2 scripts/gpt_oss/precompute_teacher_outputs.py \
    --teacher-model /path/to/gpt-oss-120b \
    --output-dir teacher_outputs/ \
    --num-steps 100000 \
    --batch-size 4 \
    --seq-len 1024

# Then train with DDP (automatically uses all precomputed steps)
accelerate launch --num_processes=2 scripts/gpt_oss/train_offline.py \
    --precomputed-teacher teacher_outputs/ \
    --student-model /path/to/gpt-oss-120b-nexus \
    --freeze-router \
    --gradient-accumulation-steps 8 \
    --output-dir outputs/
# Note: Training steps = 100,000 ÷ (8 × 2) = 6,250 optimizer steps

# 5. Validate
python scripts/gpt_oss/validate.py \
    --model outputs/checkpoint-1000 \
    --baseline /path/to/gpt-oss-120b \
    --compute-perplexity \
    --chat
```

### Testing (Quick Iteration)
```bash
# Use smaller parameters for rapid testing
python scripts/gpt_oss/collect_router_probs.py \
    --target-tokens 10000 \
    --max-candidates 1000 \
    ...

python scripts/gpt_oss/train.py \
    --max-steps 100 \
    ...
```

### Visualization
```bash
python scripts/gpt_oss/plot_metrics.py \
    --log-file outputs/training.log \
    --output plots/ \
    --smooth-window 50
```

## Key Implementation Files

### Core Architecture
**nexus/models/gpt_oss/modeling.py** (73 lines)
- `GptOssSharedExpert` class - the core shared expert implementation
- Custom SwiGLU activation matching GPT-OSS: `(up + 1) * (gate * sigmoid(gate * alpha))`
- Parameters: `alpha=1.702`, `limit=7.0` for clamping
- Three-layer MLP: gate_proj, up_proj, down_proj

### Main Scripts (9,622 total lines across 14 files)
**Training Scripts**:
- `train.py` (1,271 LOC) - Online distillation (teacher on GPU 0, student on GPU 1)
- `train_offline.py` (1,446 LOC) - Offline distillation with DDP (precomputed teacher)

**Preprocessing**:
- `collect_router_probs.py` (688 LOC) - Router statistics with distribution-aware sampling
- `analyze_pca.py` (662 LOC) - PCA analysis and expert selection
- `convert.py` (1,443 LOC) - Model conversion with PCA-guided initialization
- `precompute_teacher_outputs.py` (542 LOC) - Generate precomputed teacher outputs (with DDP support)

**Shared Modules** (new after refactoring):
- `distillation.py` (302 LOC) - KL divergence and distillation utilities
- `sfa.py` (274 LOC) - Sequential Fine-tuning with Averaging (memory-efficient merging)
- `model_utils.py` (263 LOC) - Model loading, freezing, device management

**Supporting**:
- `dataset.py` (580 LOC) - Nemotron dataset loading and token distribution analysis
- `scheduler.py` (517 LOC) - DeepSeek-V3 inspired scheduler (aux losses, warmup, cosine)
- `validate.py` (615 LOC) - Perplexity evaluation and interactive chat
- `plot_metrics.py` (774 LOC) - Training visualization
- `precomputed_loader.py` (245 LOC) - Load precomputed outputs from Parquet

### Recent Major Refactoring (Nov 2025)
The training pipeline was split from one monolithic file into two focused scripts:
- **Before**: Single train.py (2,167 lines) handling both online and offline modes
- **After**: train.py (1,271 lines) + train_offline.py (1,446 lines) + 3 shared modules (839 lines)
- **Result**: Zero code duplication, clearer separation of concerns, DDP support in precompute

## Code Architecture Insights

### Import Structure
All scripts use the NEXUS package:
```python
from nexus.models.gpt_oss import GptOssSharedExpert
```

**Important**: This is a standalone repository - all symlinks have been replaced with actual files.

### Initialization Strategies (from convert.py)
1. **pca_top24** (RECOMMENDED): PCA-guided averaging of top-24 diverse experts
2. **random**: Random initialization with proper scaling
3. **top1**: Copy most-activated expert per layer
4. **top1_average**: Copy globally most-activated expert to all layers

### Router Configurations
- **Frozen Router** (`--freeze-router`): Simpler, more stable, shared expert "fills gaps"
- **Trainable Router** (`--use-advanced-scheduler`): More flexible, needs careful tuning

### Hardware Requirements

**GPT-OSS 120B**:
- Conversion: 250GB CPU RAM, 75GB GPU VRAM
- Training: 2×98GB GPUs
- Trainable params: ~900M (0.75% of 120B total)

**GPT-OSS 20B**:
- Conversion: 150GB CPU RAM, 40GB GPU VRAM
- Training: 2×48GB GPUs (fits on A40/A6000)
- Trainable params: ~600M (3% of 20B total)

### Custom Activation Function
GPT-OSS uses a specialized SwiGLU variant that must be preserved:
```python
# Apply clamping
gate = gate.clamp(max=7.0)
up = up.clamp(-7.0, 7.0)

# Custom SwiGLU
glu = gate * torch.sigmoid(gate * 1.702)
gated_output = (up + 1) * glu

# Down projection
output = down_proj(gated_output)
```

## Dependencies and Environment

### Required Fork
GPT-OSS support requires a forked transformers library with:
- `GptOssSharedExpert` support in MoE layers (~100 lines added)
- `Mxfp4Config(dequantize=True)` for weight extraction

See README.md section "Installation" for fork details (docs/transformers_modifications.md was deleted).

### Python Dependencies
Core: torch, transformers, accelerate, datasets, scikit-learn, numpy, tqdm, matplotlib, safetensors
Optional: triton (for MXFP4 support)

Install with:
```bash
pip install -e .  # Installs from setup.py
```

## Debugging and Troubleshooting

### Training Not Converging
1. Check learning rate (likely too high)
2. Verify PCA initialization was used (not random)
3. Check teacher-student models match architecture
4. Try `--freeze-router` if router is destabilizing
5. Plot metrics: `python scripts/gpt_oss/plot_metrics.py --log-file outputs/training.log`

### Expected Training Metrics
- KL divergence: 0.06-0.08 (lower is better)
- Router drift (L1 norm): ~0.26 is normal
- Gradient norms: <0.5 is excellent
- Perplexity: Should approach teacher model after ~10K steps

### Memory Issues
- Conversion: Requires 250GB RAM (no workaround currently)
- Training OOM: Reduce batch size or use gradient accumulation
- Check GPU utilization: `nvidia-smi`
- Offline mode saves ~80GB vs online mode

### Import Errors
- Verify transformers fork is installed
- Check NEXUS package installed: `pip install -e .`
- Ensure correct conda/venv environment

### Router Collection Slow
- Increase parallelization: `--num-processes 32`
- Reduce dataset size: `--max-candidates 1000`
- Reduce target tokens: `--target-tokens 100000`

## Important Context for AI Agents

### Recent Changes (Nov 2025)
1. **Scripts are now local** - All symlinks replaced with actual file copies
2. **Training split** - Separate train.py (online) and train_offline.py (offline)
3. **Shared modules created** - distillation.py, sfa.py, model_utils.py
4. **DDP support added to precompute** - 2x faster teacher output generation
5. **Dataset exhaustion handling** - Clear error messages, no silent failures

### Project Philosophy
- **Ship working code first, polish later**: Focus on functional implementation
- **Extensibility over generality**: Each model family gets its own namespace
- **Research quality**: Demonstrates efficient MoE adaptation, not production-hardened
- **Priority order**: GPT-OSS (v0.1.0) → Testing/polish (v0.1.1) → New families (v0.2.0+)

### When Making Changes
1. **Always read files first** - Never propose changes to unread code
2. **Preserve existing style** - Match indentation, naming, patterns
3. **Consider memory** - Large model changes affect hardware requirements
4. **Update documentation** - Keep docs in sync with code
5. **Test incrementally** - Verify changes don't break existing functionality

### Known Limitations
1. Conversion requires 250GB RAM (limiting factor for many users)
2. Currently GPT-OSS only (Llama-MoE, Qwen-MoE planned for v0.2.0+)
3. Requires transformers fork (upstreaming changes TODO)
4. Tests not yet implemented (see TODO.md)
5. Print statements instead of proper logging (polish item for v0.1.1)

### Code Quality Notes
- Type hints: Partial coverage (improving gradually)
- Documentation: Good docstrings in main modules
- Error handling: Basic, could be more informative
- Logging: Currently uses print() - proper logging is TODO

## Adding New Model Families

To add support for a new MoE architecture (e.g., Llama-MoE):

1. Create `nexus/models/llama_moe/` directory
2. Implement `LlamaMoESharedExpert` class (similar to GptOssSharedExpert)
3. Create `scripts/llama_moe/` with family-specific scripts
4. Adapt router collection for Llama's routing format
5. Implement conversion logic for Llama architecture
6. Test convergence and perplexity
7. Update documentation

See TODO.md for planned model families.

## References

**Essential Documentation**:
- `README.md` - Project overview and quick start
- `TODO.md` - Development roadmap and current status
- `scripts/gpt_oss/README.md` - Detailed script documentation
- `docs/architecture.md` - Design decisions and innovations
- `docs/quickstart.md` - Step-by-step tutorial

**Key Research**:
- Sequential Fine-tuning with Averaging (SFA): arXiv:2501.05559
- Knowledge Distillation: Hinton et al., 2015
- DeepSeek-V3 for scheduler inspiration

## Quick Decision Matrix

**Use `train.py` (online) when**:
- Quick experiments or debugging
- Training <5K steps
- Have 2 GPUs available
- Don't want preprocessing overhead

**Use `train_offline.py` (offline) when**:
- Production training runs (10K+ steps)
- Have 2+ GPUs for DDP
- Want maximum training speed
- Can precompute once, reuse many times

**Freeze router when**:
- First experiments with new data
- Want stable, predictable training
- Shared expert should "fill gaps"

**Train router when**:
- Domain specialization needed
- Comfortable with advanced tuning
- Using DeepSeek-V3 scheduler
