# GPT-OSS Scripts for NEXUS

This directory contains the complete workflow implementation for GPT-OSS models in NEXUS.

## Status

✅ **All scripts are now local copies** - No external dependencies on gpt-oss-shared
✅ **Refactored training scripts** - Separate online and offline modes for clarity
✅ **Supports both GPT-OSS 20B and 120B** - Model variant auto-detected from config

The scripts implement the full NEXUS pipeline for adding and training shared experts on GPT-OSS MoE models.

## Available Scripts

### Core Pipeline

**collect_router_probs.py** (685 LOC)
- Collects router activation statistics on diverse data
- Uses distribution-aware sampling for vocabulary coverage
- Parallelized across multiple CPU cores
- Generates `.npz` files with router probabilities

**analyze_pca.py** (659 LOC)
- Performs PCA analysis on router activations
- Identifies most important and diverse experts
- Generates expert selection statistics for initialization
- Creates visualization plots

**convert.py** (1,443 LOC)
- Transforms standard GPT-OSS model into 4+1 architecture
- Adds shared expert to each MoE layer
- Supports multiple initialization strategies (PCA-guided recommended)
- Preserves MXFP4 quantization for frozen experts

### Training Scripts (CHOOSE ONE)

**train.py** (1,271 LOC) - **Online Distillation**
- Teacher model loaded on GPU 0, student on GPU 1
- Teacher forward pass computed inline during training
- Simple gradient accumulation (no DDP overhead)
- Best for: Quick iteration, debugging, small-scale experiments
- Hardware: 2 GPUs (one per model)

**train_offline.py** (1,446 LOC) - **Offline Distillation with DDP**
- Uses precomputed teacher outputs (teacher NOT in memory)
- Multi-GPU training with Distributed Data Parallel
- Saves ~80GB VRAM by not loading teacher
- Best for: Production training, faster training speed
- Hardware: 2+ GPUs for student model DDP
- Requires: Precomputed teacher outputs (run `precompute_teacher_outputs.py` first)

### Preprocessing for Offline Training

**precompute_teacher_outputs.py** (525 LOC)
- Generates precomputed teacher outputs for offline distillation
- Saves teacher forward pass results to Parquet files
- **DDP support**: Use 2+ GPUs for faster precomputation (2x speedup)
- **Error handling**: Errors out with clear message if dataset exhausted
- **Metadata**: Saves actual steps completed (even on error)
- Storage: ~15-25 GB for 10K steps (batch=4, seq=1024)
- Run once, use for many training runs

**precomputed_loader.py** (242 LOC)
- Efficiently loads precomputed teacher outputs from Parquet files
- Used automatically by train_offline.py
- Supports top-K logit compression for smaller storage

### Validation

**validate.py** (612 LOC)
- Evaluates model perplexity vs baseline
- Interactive chat mode for qualitative testing
- Compares student and teacher outputs

### Shared Modules (NEW!)

**distillation.py** (302 LOC)
- Knowledge distillation utilities
- Temperature-scaled KL divergence computation
- Router logits validation
- Supports both online and offline modes

**sfa.py** (274 LOC)
- SFA (Sequential Fine-tuning with Averaging) implementation
- Memory-efficient checkpoint loading (~900MB vs 60GB)
- In-place merging with anchor checkpoints
- DDP-safe broadcasting (for train_offline.py)

**model_utils.py** (263 LOC)
- Model loading, freezing, and configuration
- Device validation and parsing
- Stage-0 preparation (freeze experts, attention, embeddings)
- Gradient checkpointing setup

### Supporting Modules

**dataset.py** (580 LOC)
- Dataset loading and preparation
- Supports multiple Nemotron instruction datasets
- Streaming and caching support
- Token distribution analysis

**scheduler.py** (517 LOC)
- DeepSeek-V3 inspired advanced scheduler
- Handles aux losses (load balancing, router z-loss)
- Stage-based learning rate scheduling
- Warmup and cosine decay

**plot_metrics.py** (771 LOC)
- Visualizes training metrics from logs
- Generates loss curves, KL divergence plots
- Router drift analysis
- Gradient norm tracking

## Choosing a Training Script

### train.py vs train_offline.py

**Use `train.py` (Online Distillation) when:**
- ✅ You have 2 GPUs and want simple setup
- ✅ Quick iteration / debugging / experimentation
- ✅ Don't want to precompute teacher outputs
- ✅ Training for <5K steps
- ❌ Slightly slower (teacher forward pass overhead)

**Use `train_offline.py` (Offline Distillation with DDP) when:**
- ✅ You have 2+ GPUs and want maximum speed
- ✅ Production training runs
- ✅ Can precompute teacher outputs once, reuse many times
- ✅ Want DDP for faster training
- ✅ Want to save ~80GB VRAM (no teacher in memory)
- ✅ Simple configuration (only 3 required arguments!)
- ❌ Requires preprocessing step (precompute teacher)

**Summary**: For quick experiments, use `train.py`. For production, use `train_offline.py`.

---

## Quick Start

### Complete Workflow

```bash
# 1. Collect router statistics (~4 hours for 1M tokens)
python collect_router_probs.py \
    --model /path/to/gpt-oss-120b \
    --target-tokens 1000000 \
    --output data/router_probs.npz

# 2. PCA analysis (~5 min)
python analyze_pca.py \
    --input data/router_probs.npz \
    --output data/pca_stats.json \
    --top-k 24

# 3. Convert model (~15 min, requires 250GB RAM)
python convert.py \
    --input /path/to/gpt-oss-120b \
    --output /path/to/gpt-oss-120b-nexus \
    --pca-stats data/pca_stats.json

# 4. Train shared expert - OPTION A: Online distillation (simple, 2 GPUs)
python train.py \
    --teacher-model /path/to/gpt-oss-120b \
    --student-model /path/to/gpt-oss-120b-nexus \
    --teacher-device cuda:0 \
    --student-device cuda:1 \
    --freeze-router \
    --max-steps 1000 \
    --output-dir outputs/trained

# 4. Train shared expert - OPTION B: Offline distillation (DDP, 2+ GPUs, faster)
# B1. Precompute teacher outputs (run once)
# Single GPU (slower):
python precompute_teacher_outputs.py \
    --teacher-model /path/to/gpt-oss-120b \
    --output-dir teacher_outputs/ \
    --num-steps 100000 \
    --batch-size 4 \
    --seq-len 1024

# OR Multi-GPU with DDP (2x faster):
accelerate launch --num_processes=2 precompute_teacher_outputs.py \
    --teacher-model /path/to/gpt-oss-120b \
    --output-dir teacher_outputs/ \
    --num-steps 100000 \
    --batch-size 4 \
    --seq-len 1024

# B2. Train with DDP (automatically uses all precomputed steps)
accelerate launch --num_processes=2 train_offline.py \
    --precomputed-teacher teacher_outputs/ \
    --student-model /path/to/gpt-oss-120b-nexus \
    --freeze-router \
    --gradient-accumulation-steps 8 \
    --output-dir outputs/trained
# Note: Training steps automatically calculated from precomputed data
# Example: 100,000 steps ÷ (8 × 2) = 6,250 optimizer steps

# 5. Validate results
python validate.py \
    --model outputs/trained/checkpoint-1000 \
    --baseline /path/to/gpt-oss-120b \
    --compute-perplexity \
    --chat
```

### Fast Testing (Quick Iteration)

```bash
# Use smaller parameters for testing
python collect_router_probs.py \
    --target-tokens 10000 \
    --max-candidates 1000 \
    ...

python train.py \
    --max-steps 100 \
    ...
```

## Implementation Details

### Import Structure

All scripts now use the NEXUS package for shared expert implementation:

```python
from nexus.models.gpt_oss import GptOssSharedExpert
```

**Installation**: Ensure NEXUS package is installed:
```bash
cd /path/to/nexus
pip install -e .
```

### Initialization Strategies

The conversion script supports multiple strategies (from convert.py):

1. **pca_top24** (RECOMMENDED): PCA-guided averaging of top-24 diverse experts
2. **random**: Random initialization with proper scaling
3. **top1**: Copy most-activated expert per layer
4. **top1_average**: Copy globally most-activated expert to all layers

### Hardware Requirements

**Conversion**:
- CPU RAM: 250GB (for PCA weight computation)
- GPU VRAM: 75GB (for model assembly)
- Time: ~15-20 minutes

**Training**:
- GPU VRAM: 2×98GB (DDP across consumer GPUs)
- Time: ~2-3 hours for 1000 steps
- Trainable params: ~900M (0.75% of 120B total)

### Router Configurations

**Frozen Router** (--freeze-router):
- Simpler optimization
- More stable training
- Shared expert learns to "fill gaps"
- Recommended for initial experiments

**Trainable Router** (--use-advanced-scheduler):
- More flexible adaptation
- Requires careful tuning
- Uses DeepSeek-V3 scheduler
- Good for domain specialization

## Dataset Support

Currently supports Nemotron instruction datasets:
- `nemotron-general`: General instructions
- `nemotron-code`: Code-focused tasks
- `nemotron-math`: Mathematical reasoning
- `nemotron-tool-calling`: Tool use instructions

Default paths can be overridden with command-line arguments.

## Output Files

**collect_router_probs.py**:
- `router_probs.npz`: Router activation probabilities per layer/expert

**analyze_pca.py**:
- `pca_stats.json`: Expert selection statistics
- `pca_plots/`: Visualization of expert importance

**convert.py**:
- Modified model directory with shared experts added

**train.py**:
- `checkpoint-*/`: Model checkpoints
- `training.log`: Detailed training metrics

**plot_metrics.py**:
- `plots/*.png`: Training curve visualizations

## Troubleshooting

**Import errors**: Ensure NEXUS package is installed (`pip install -e .`)

**Memory errors during conversion**: Requires 250GB RAM for PCA computation

**Memory errors during training**: Reduce batch size or use gradient accumulation

**Training not converging**: Check learning rate, verify PCA initialization was used

**Slow router collection**: Increase `--num-processes` for more parallelization

## Advanced Usage

### Custom Datasets

See `dataset.py` for adding custom dataset sources.

### Offline Distillation

Precompute teacher outputs for faster training:
```bash
# Note: precompute_teacher.py needs to be added to this directory
```

### Visualization

```bash
python plot_metrics.py \
    --log-file outputs/training.log \
    --output plots/ \
    --smooth-window 50
```

## Architecture Notes

These scripts are **GPT-OSS specific**. Future model families (Llama-MoE, Qwen-MoE) will have their own subdirectories with family-specific adaptations.

All scripts follow the NEXUS philosophy: efficient MoE adaptation through trainable shared experts.

## References

For more details:
- Project overview: `/nexus/README.md`
- Architecture deep dive: `/nexus/docs/architecture.md`
- Transformers fork requirements: `/nexus/docs/transformers_modifications.md`
- Development guide: `/nexus/claude.md` (for contributors)
