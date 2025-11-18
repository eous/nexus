#!/usr/bin/env python3
"""
Complete NEXUS workflow for GPT-OSS models.

This example demonstrates the full pipeline:
1. Collect router statistics
2. PCA analysis
3. Model conversion
4. Training
5. Validation

Usage:
    python examples/gpt_oss_workflow.py --model-path /path/to/gpt-oss-120b
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run shell command with error handling."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n✗ Failed: {description}")
        sys.exit(1)

    print(f"\n✓ Completed: {description}")


def main():
    parser = argparse.ArgumentParser(description="Complete NEXUS workflow for GPT-OSS")
    parser.add_argument("--model-path", required=True, help="Path to base GPT-OSS model")
    parser.add_argument("--output-dir", default="nexus_output", help="Output directory")
    parser.add_argument("--target-tokens", type=int, default=100000, help="Tokens for PCA (100K for testing, 1M for production)")
    parser.add_argument("--top-k", type=int, default=24, help="Number of experts to select")
    parser.add_argument("--training-steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--skip-collection", action="store_true", help="Skip router probability collection (reuse existing)")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip model conversion (reuse existing)")

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    model_output = output_dir / "model_nexus"
    training_output = output_dir / "trained"

    print("="*80)
    print("NEXUS Workflow for GPT-OSS")
    print("="*80)
    print(f"Base model: {args.model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target tokens: {args.target_tokens:,}")
    print(f"Top-K experts: {args.top_k}")
    print(f"Training steps: {args.training_steps}")
    print("="*80)

    # Step 1: Collect router probabilities
    if not args.skip_collection:
        run_command([
            "python", "scripts/gpt_oss/collect_router_probs.py",
            "--model", args.model_path,
            "--target-tokens", str(args.target_tokens),
            "--dataset", "nemotron-mixed",
            "--output", str(data_dir / "router_probs.npz"),
        ], "Step 1: Collect Router Probabilities")
    else:
        print("\n✓ Skipping router probability collection (reusing existing)")

    # Step 2: PCA analysis
    run_command([
        "python", "scripts/gpt_oss/analyze_pca.py",
        "--input", str(data_dir / "router_probs.npz"),
        "--output", str(data_dir / "pca_stats.json"),
        "--top-k", str(args.top_k),
        "--plot-dir", str(output_dir / "plots"),
    ], "Step 2: PCA Analysis")

    # Step 3: Model conversion
    if not args.skip_conversion:
        run_command([
            "python", "scripts/gpt_oss/convert.py",
            "--input", args.model_path,
            "--output", str(model_output),
            "--router-stats", str(data_dir / "pca_stats.json"),
            "--init-strategy", f"pca_top{args.top_k}",
        ], "Step 3: Convert Model (Add Shared Expert)")
    else:
        print("\n✓ Skipping model conversion (reusing existing)")

    # Step 4: Training
    run_command([
        "python", "scripts/gpt_oss/train.py",
        "--student-model", str(model_output),
        "--teacher-model", args.model_path,
        "--freeze-router",
        "--use-advanced-scheduler",
        "--batch-size", "1",
        "--gradient-accumulation-steps", "4",
        "--learning-rate", "1e-5",
        "--max-steps", str(args.training_steps),
        "--seq-len", "1024",
        "--output-dir", str(training_output),
    ], "Step 4: Train Shared Expert")

    # Step 5: Validation
    final_checkpoint = training_output / f"checkpoint-{args.training_steps}"

    run_command([
        "python", "scripts/validate_model.py",
        "--model", str(final_checkpoint),
        "--baseline", args.model_path,
        "--compute-perplexity",
        "--num-samples", "500",
    ], "Step 5: Validate Final Model")

    # Summary
    print("\n" + "="*80)
    print("NEXUS Workflow Complete!")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  PCA analysis: {data_dir / 'pca_stats.json'}")
    print(f"  Plots: {output_dir / 'plots'}")
    print(f"  Converted model: {model_output}")
    print(f"  Trained model: {final_checkpoint}")
    print(f"\nNext steps:")
    print(f"  - Review plots in {output_dir / 'plots'}")
    print(f"  - Test interactively: python scripts/validate_model.py --model {final_checkpoint} --chat")
    print(f"  - Compare with baseline perplexity scores")
    print("="*80)


if __name__ == "__main__":
    main()
