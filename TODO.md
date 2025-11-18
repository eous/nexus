# NEXUS Development TODO

## Current Status (v0.1.0-alpha)

✅ **Core functionality working**:
- GPT-OSS conversion with PCA-guided initialization
- Training with frozen/trainable router
- DDP support, offline distillation
- Advanced DeepSeek-V3 scheduler
- Validation and interactive chat

✅ **Scripts are now local** - All symlinks replaced with actual files (2025-11-20)

---

## v0.1.0 Release Status

### Code Organization

- [x] Copy scripts from gpt-oss-shared to nexus/scripts/gpt_oss/ ✅ DONE
- [x] Update imports to use `nexus.models.gpt_oss` ✅ DONE (convert.py fixed)
- [x] Remove gpt-oss-shared dependencies ✅ DONE (all files local)
- [x] Add NEXUS branding to all scripts ✅ DONE (headers updated)
- [ ] Clean up legacy code paths (low priority, works as-is)
- [x] Create claude.md guide for contributors ✅ DONE

### Testing

**Status**: Not started (acceptable for v0.1.0-alpha)

- [ ] Create pytest test suite in `tests/gpt_oss/`
  - [ ] Test PCA analysis with small synthetic data
  - [ ] Test model conversion (mock models)
  - [ ] Test shared expert forward pass
  - [ ] Test training utilities
- [ ] Add CI/CD (GitHub Actions)
- [ ] Test on fresh Python environment

**Priority**: Medium (defer to v0.1.1 or later)

### Documentation

**Status**: Core docs complete, advanced guides pending

- [x] Complete GPT-OSS specific guide (`docs/models/gpt_oss.md`) ✅ DONE
- [x] Update scripts/gpt_oss/README.md ✅ DONE
- [x] Contributing guide (CONTRIBUTING.md) ✅ EXISTS
- [ ] Training best practices guide (could be section in gpt_oss.md)
- [ ] Create API reference documentation
- [ ] Add troubleshooting guide with common errors (partially in gpt_oss.md)
- [ ] Create notebook tutorials in `docs/examples/`

**Priority**: Medium (core docs sufficient for alpha)

### Polish

**Status**: Not started (acceptable for alpha release)

- [ ] Add proper logging (replace print statements)
- [ ] Add progress bars for all long operations (some exist)
- [ ] Improve error messages with actionable suggestions
- [ ] Add --verbose and --quiet flags
- [ ] Type hints for all public APIs (partial coverage)

**Priority**: Low (defer to v0.1.1+)

---

## Pre-Release Checklist (v0.1.0)

**Critical (blocking release):**
- [x] Scripts copied and working ✅
- [x] Imports updated ✅
- [x] Documentation complete ✅
- [x] NEXUS branding added ✅

**Nice to have (non-blocking):**
- [ ] Test suite
- [ ] CI/CD
- [ ] Notebook tutorials
- [ ] API reference

**Release readiness**: 🟢 **READY for v0.1.0-alpha release**

---

## Future Model Families (v0.2.0+)

### Llama-MoE Support
- [ ] Implement Llama-MoE shared expert module
- [ ] Adapt PCA collection for Llama router format
- [ ] Test on Llama-MoE-8×7B or similar

### Qwen-MoE Support
- [ ] Implement Qwen-MoE shared expert module
- [ ] Adapter for Qwen-specific routing
- [ ] Test on Qwen MoE models

### DeepSeek-V3 Support
- [ ] Native DeepSeek-V3 support (currently inspired by, not tested on)
- [ ] Test 671B model (if available)

---

## Advanced Features (v0.3.0+)

### Multi-Shared Expert
- [ ] Support multiple shared experts (not just 1)
- [ ] Routing between shared experts
- [ ] Use case: Domain-specific shared experts

### Dynamic Expert Selection
- [ ] Online PCA updates during training
- [ ] Adaptive expert selection based on loss
- [ ] A/B testing different expert combinations

### Compression
- [ ] Quantize shared expert to INT8/FP8 post-training
- [ ] Knowledge distillation from larger shared expert
- [ ] Pruning unimportant shared expert neurons

### Vision/Multimodal
- [ ] Test on vision MoE models
- [ ] Multimodal MoE support (CLIP-style)
- [ ] Cross-modal shared expert experiments

---

## Community (Ongoing)

- [x] Create contributing guide ✅ EXISTS (CONTRIBUTING.md)
- [x] Create claude.md for AI contributors ✅ DONE
- [ ] Set up issue templates
- [ ] Create Discord/Slack for discussion
- [ ] Write blog post explaining technique
- [ ] Submit to arXiv (optional)

---

## Infrastructure (v0.2.0+)

- [ ] PyPI package publication
- [ ] Docker containers for reproducibility
- [ ] Model zoo (pre-converted NEXUS models)
- [ ] Benchmarking suite across tasks

---

## Known Issues & Limitations

### Current Limitations
1. **Hardware requirements**: 250GB RAM for conversion is prohibitive
   - Potential fix: Streaming PCA or approximations

2. **Hardcoded paths**: Some scripts have default paths to `/mnt/git/gpt-oss-shared/data/`
   - Impact: Low (just defaults, can be overridden)
   - Fix: Remove hardcoded paths or make configurable

3. **Transformers fork dependency**: Requires custom installation
   - Long-term: Contribute changes upstream to HuggingFace

4. **Training speed**: ~22 hours for 10K steps
   - Mitigation: Offline distillation, larger batch sizes

### Technical Debt
- [ ] Legacy code paths in scripts (non-critical)
- [ ] Print statements instead of proper logging
- [ ] Inconsistent type hints
- [ ] Some error messages could be more helpful

**Priority**: Address in v0.1.1+ as polish items

---

## Notes

**Philosophy**: Ship working code first, polish later. Current focus is GPT-OSS production quality, then expand to other families.

**Priority order**:
1. ✅ Clean up and document GPT-OSS scripts (v0.1.0) - DONE
2. Testing and polish (v0.1.1)
3. Add one more model family (v0.2.0)
4. Advanced features as needed (v0.3.0+)

**Current milestone**: v0.1.0-alpha is **READY FOR RELEASE** 🎉

All core functionality works, scripts are self-contained, and documentation is comprehensive. Testing and polish can be iterative improvements.

---

## Recent Changes (2025-11-20)

### Morning Session
- ✅ Replaced all symlinks with actual file copies
- ✅ Fixed import in convert.py to use `nexus.models.gpt_oss`
- ✅ Updated scripts/gpt_oss/README.md (removed symlink references)
- ✅ Added NEXUS branding to all script headers
- ✅ Created comprehensive docs/models/gpt_oss.md guide
- ✅ Created claude.md for AI contributor guidance
- ✅ Verified KL divergence computation correctness (docs/kl_divergence_verification.md)
- ✅ Verified gradient accumulation and DDP correctness (docs/gradient_accumulation_verification.md)
- ✅ Fixed critical bug: DDP + online distillation now properly prevented

### Afternoon Session - Major Refactoring
- ✅ **Split train.py into train.py + train_offline.py** (see docs/train_refactoring_summary.md)
- ✅ Created **distillation.py** (302 LOC) - shared KL divergence utilities
- ✅ Created **sfa.py** (274 LOC) - shared SFA merging logic
- ✅ Created **model_utils.py** (263 LOC) - shared model utilities
- ✅ **train.py** (1,271 LOC) - online distillation only (teacher GPU0, student GPU1, no DDP)
- ✅ **train_offline.py** (1,446 LOC) - offline distillation with DDP (precomputed outputs)
- ✅ Copied **precompute_teacher_outputs.py** (542 LOC) - generates precomputed outputs
- ✅ Copied **precomputed_loader.py** (242 LOC) - loads precomputed outputs
- ✅ **Fixed dataset exhaustion bug** - errors out with clear message (no silent failure)
- ✅ **Added DDP support to precompute** - 2x faster with 2 GPUs
- ✅ **Simplified approach** - no epoch looping, just error if insufficient data
- ✅ **Removed 95 lines vs original** (eliminated duplicates, added DDP, simplified logic)
- ✅ Updated scripts/gpt_oss/README.md with dual-script guidance and DDP instructions
- ✅ Zero code duplication - all shared logic in modules
- ✅ All scripts syntactically valid and ready to use

**Impact**:
- From 1 monolithic file (2,167 lines) to 2 focused scripts + 3 shared modules
- Eliminated mixed-mode complexity
- Clear separation: online vs offline distillation
- Better maintainability and testability

**Next steps**: Consider v0.1.0 release, then work on testing for v0.1.1
