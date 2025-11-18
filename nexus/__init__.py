"""
NEXUS: Neural Expert Unified Specialization

Efficient fine-tuning for Mixture-of-Experts models on minimal hardware.
"""

__version__ = "0.1.0"

from nexus.models import gpt_oss

__all__ = ["gpt_oss", "__version__"]
