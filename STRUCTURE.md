# NEXUS Directory Structure

```
nexus/
├── README.md                       # Project overview
├── GETTING_STARTED.md              # 5-minute quickstart
├── PROJECT_STATUS.md               # Current implementation status
├── TODO.md                         # Development roadmap
├── CONTRIBUTING.md                 # Contribution guide
├── LICENSE                         # MIT license
├── .gitignore                      # Git ignore rules
├── setup.py                        # Package configuration
├── requirements.txt                # Dependencies
│
├── nexus/                          # Python package
│   ├── __init__.py                 # Package root (v0.1.0)
│   ├── core/                       # Model-agnostic utilities
│   │   └── __init__.py             # (PCA, training utils - TODO)
│   ├── models/                     # Model family implementations
│   │   ├── __init__.py
│   │   └── gpt_oss/                # GPT-OSS support
│   │       ├── __init__.py
│   │       └── modeling.py         # GptOssSharedExpert
│   └── utils/                      # Common utilities
│       └── __init__.py
│
├── scripts/                        # Executable scripts
│   └── gpt_oss/                    # GPT-OSS workflow
│       ├── README.md               # Script documentation
│       ├── collect_router_probs.py # Symlink → /mnt/git/gpt-oss-shared/...
│       ├── analyze_pca.py          # Symlink → ...
│       ├── convert.py              # Symlink → ...
│       ├── train.py                # Symlink → ...
│       ├── validate.py             # Symlink → ...
│       ├── plot_metrics.py         # Symlink → ...
│       ├── scheduler.py            # Symlink → ...
│       └── dataset.py              # Symlink → ...
│
├── examples/                       # Usage examples
│   └── gpt_oss_workflow.py         # Complete pipeline automation
│
├── docs/                           # Documentation
│   ├── quickstart.md               # Detailed workflow guide
│   ├── architecture.md             # Design decisions & comparisons
│   ├── transformers_modifications.md # Fork requirements
│   ├── models/                     # Model-specific guides (TODO)
│   └── examples/                   # Notebooks (TODO)
│
└── tests/                          # Test suite (TODO)
    └── gpt_oss/                    # GPT-OSS tests
```

## Quick Navigation

### For Users
- **Start here**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **Learn more**: [docs/quickstart.md](docs/quickstart.md)
- **Understand design**: [docs/architecture.md](docs/architecture.md)

### For Developers
- **Status**: [PROJECT_STATUS.md](PROJECT_STATUS.md)
- **Roadmap**: [TODO.md](TODO.md)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Fork details**: [docs/transformers_modifications.md](docs/transformers_modifications.md)

### For Running
- **Scripts**: `scripts/gpt_oss/*.py`
- **Example**: `examples/gpt_oss_workflow.py`
- **Package code**: `nexus/models/gpt_oss/`

## File Counts

- **Documentation**: 8 files (README, guides, architecture)
- **Python package**: 7 files (package structure + modeling)
- **Scripts**: 9 files (working implementation via symlinks)
- **Examples**: 1 file (end-to-end workflow)
- **Config**: 4 files (setup.py, requirements, license, gitignore)

**Total**: 29 files, fully functional and documented
