# Contributing to NEXUS

Thank you for your interest in NEXUS! We welcome contributions of all kinds.

## Ways to Contribute

### 1. Add Support for New Model Families

NEXUS is designed to be extensible to any MoE architecture. To add a new family:

**Required files** (using Llama-MoE as example):
```
nexus/models/llama_moe/
├── __init__.py
├── modeling.py          # LlamaMoESharedExpert class
└── conversion.py        # Family-specific conversion logic

scripts/llama_moe/
├── collect_router_probs.py   # May need router format adaptation
├── analyze_pca.py            # Usually works as-is
├── convert.py                # Family-specific
└── train.py                  # May need loss function adaptation
```

**Key implementation**:
```python
# nexus/models/llama_moe/modeling.py

class LlamaMoESharedExpert(nn.Module):
    """Shared expert for Llama-MoE models."""

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        # Implement using Llama's activation (SiLU typically)
        # Match routed expert structure
        ...

    def forward(self, hidden_states):
        # Always-active computation for all tokens
        ...
```

**Testing checklist**:
- [ ] Shared expert forward pass works
- [ ] Can be attached to MoE layer
- [ ] Training converges
- [ ] Perplexity degradation <5%

### 2. Improve PCA Analysis

**Ideas**:
- Multi-component weighting (not just first PC)
- Entropy-guided expert selection
- Per-layer adaptive top-K
- Cross-layer expert clustering

**Location**: `nexus/core/pca.py` (to be created)

### 3. Add Training Features

**Ideas**:
- Learnable `shared_expert_scale` (vs fixed 0.2)
- Token-dependent gating
- Multi-task training
- Curriculum learning for shared expert

**Location**: Training scripts or `nexus/core/training.py`

### 4. Optimize Performance

**Ideas**:
- Faster PCA with approximate methods
- Batched expert averaging
- Gradient checkpointing for shared expert
- Mixed precision training improvements

### 5. Improve Documentation

**Needs**:
- Jupyter notebook tutorials
- Video walkthroughs
- Blog posts explaining technique
- Benchmark results across tasks

## Development Setup

```bash
# Clone with development dependencies
git clone https://github.com/yourusername/nexus.git
cd nexus
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black nexus/ scripts/ tests/

# Lint
flake8 nexus/ scripts/
```

## Code Style

- **Python**: Follow PEP 8, use Black formatter
- **Docstrings**: Google style
- **Type hints**: Required for all public APIs
- **Comments**: Explain "why", not "what"

## Pull Request Process

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Make changes** with tests
4. **Run tests**: `pytest tests/`
5. **Format code**: `black .`
6. **Submit PR** with clear description

**PR template**:
```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Added X
- Modified Y
- Fixed Z

## Testing
- [ ] Added tests
- [ ] All tests pass
- [ ] Tested on real model

## Checklist
- [ ] Code formatted with Black
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## Questions?

- Open an issue for bugs/features
- Discussions for design questions
- Discord (TBD) for community chat

## License

By contributing, you agree your contributions will be licensed under the MIT License.
