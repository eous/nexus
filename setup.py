"""
NEXUS: Neural Expert Unified Specialization
Efficient fine-tuning for Mixture-of-Experts models
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="nexus-moe",
    version="0.1.0",
    description="Efficient fine-tuning for Mixture-of-Experts models on minimal hardware",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/nexus",
    packages=find_packages(exclude=["tests", "scripts", "examples", "docs"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "safetensors>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="mixture-of-experts moe fine-tuning parameter-efficient pca neural-networks",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/nexus/issues",
        "Source": "https://github.com/yourusername/nexus",
        "Documentation": "https://nexus.readthedocs.io",
    },
)
