#!/usr/bin/env python3
"""
NEXUS: Neural Expert Unified Specialization
GPT-OSS PCA Analysis

Analyze router probabilities using PCA to select diverse experts.

This script performs Principal Component Analysis on router probability
distributions to identify the most important and diverse experts for
merging into a shared expert.

Usage:
    python analyze_router_pca.py \
        --input data/router_probs.npz \
        --output data/pca_stats.json \
        --top-k 24 \
        --plot-dir plots/pca/

Requirements:
    pip install scikit-learn matplotlib
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def perform_pca_analysis(
    router_probs: np.ndarray,
    n_components: int = 20,
) -> List[Dict]:
    """
    Perform PCA on router probabilities for each layer.

    Args:
        router_probs: Array of shape [num_tokens, num_layers, num_experts]
        n_components: Number of principal components to compute

    Returns:
        List of per-layer PCA results
    """
    num_tokens, num_layers, num_experts = router_probs.shape
    print(f"Performing PCA analysis on {num_layers} layers...")
    print(f"Input shape: {router_probs.shape}")

    layer_results = []

    for layer_idx in range(num_layers):
        # Extract probabilities for this layer: [num_tokens, num_experts]
        layer_probs = router_probs[:, layer_idx, :]

        # Perform PCA
        n_comp = min(n_components, num_tokens, num_experts)
        pca = PCA(n_components=n_comp)
        pca.fit(layer_probs)

        # Calculate expert importance scores
        # Importance = sum of absolute contributions weighted by variance explained
        expert_importance = np.zeros(num_experts)
        for i in range(n_comp):
            component = pca.components_[i]
            variance_ratio = pca.explained_variance_ratio_[i]
            expert_importance += np.abs(component) * variance_ratio

        layer_results.append({
            "layer_idx": layer_idx,
            "pca": pca,
            "expert_importance": expert_importance,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "components": pca.components_,
        })

        # Print summary for this layer
        cumulative_var = np.cumsum(pca.explained_variance_ratio_)
        var_90 = np.argmax(cumulative_var >= 0.90) + 1 if cumulative_var[-1] >= 0.90 else n_comp
        print(f"  Layer {layer_idx:2d}: "
              f"Top-5 components explain {cumulative_var[4]:.1%}, "
              f"{var_90} components for 90%")

    return layer_results


def analyze_expert_distribution(
    layer_results: List[Dict],
) -> Dict:
    """
    Analyze the distribution of expert importance scores.

    Args:
        layer_results: List of per-layer PCA results

    Returns:
        Dictionary with distribution statistics
    """
    print(f"\nAnalyzing expert importance distribution...")

    distribution_stats = {
        'layers': {},
        'global': {
            'all_scores': [],
        }
    }

    for result in layer_results:
        layer_idx = result["layer_idx"]
        importance = result["expert_importance"]

        # Sort in descending order
        sorted_importance = np.sort(importance)[::-1]

        # Calculate statistics
        total_importance = importance.sum()
        top_4_importance = sorted_importance[:4].sum()
        top_8_importance = sorted_importance[:8].sum()
        top_16_importance = sorted_importance[:16].sum()
        top_32_importance = sorted_importance[:32].sum()

        # Percentiles
        percentiles = {
            'p50': float(np.percentile(importance, 50)),
            'p75': float(np.percentile(importance, 75)),
            'p90': float(np.percentile(importance, 90)),
            'p95': float(np.percentile(importance, 95)),
            'p99': float(np.percentile(importance, 99)),
        }

        # Check for cliff (large gap between ranks)
        gaps = np.diff(sorted_importance)
        max_gap_idx = np.argmax(np.abs(gaps))
        max_gap = float(gaps[max_gap_idx])

        # Use string keys for consistency with save_pca_stats
        layer_key = f"layer_{layer_idx}"
        distribution_stats['layers'][layer_key] = {
            'sorted_importance': sorted_importance.tolist(),
            'top_4_fraction': float(top_4_importance / total_importance),
            'top_8_fraction': float(top_8_importance / total_importance),
            'top_16_fraction': float(top_16_importance / total_importance),
            'top_32_fraction': float(top_32_importance / total_importance),
            'percentiles': percentiles,
            'max_gap_idx': int(max_gap_idx),
            'max_gap_value': max_gap,
            'mean': float(importance.mean()),
            'std': float(importance.std()),
        }

        distribution_stats['global']['all_scores'].extend(importance.tolist())

        # Print summary for this layer
        print(f"  Layer {layer_idx:2d}: "
              f"Top-4={top_4_importance/total_importance:.1%}, "
              f"Top-8={top_8_importance/total_importance:.1%}, "
              f"Top-16={top_16_importance/total_importance:.1%}, "
              f"Max gap at rank {max_gap_idx+1}")

    # Global statistics
    all_scores = np.array(distribution_stats['global']['all_scores'])
    distribution_stats['global']['mean'] = float(all_scores.mean())
    distribution_stats['global']['std'] = float(all_scores.std())
    distribution_stats['global']['percentiles'] = {
        'p50': float(np.percentile(all_scores, 50)),
        'p75': float(np.percentile(all_scores, 75)),
        'p90': float(np.percentile(all_scores, 90)),
        'p95': float(np.percentile(all_scores, 95)),
        'p99': float(np.percentile(all_scores, 99)),
    }

    return distribution_stats


def select_top_k_experts(
    layer_results: List[Dict],
    top_k: int = 4,
) -> Dict:
    """
    Select top-K most important experts per layer based on PCA importance.

    Args:
        layer_results: List of per-layer PCA results
        top_k: Number of experts to select per layer

    Returns:
        Dictionary mapping layer indices to top-K expert indices
    """
    print(f"\nSelecting top-{top_k} experts per layer...")

    selection = {}

    for result in layer_results:
        layer_idx = result["layer_idx"]
        importance = result["expert_importance"]

        # Get top-K expert indices
        top_indices = np.argsort(importance)[-top_k:][::-1]  # Descending order
        top_scores = importance[top_indices]

        selection[layer_idx] = {
            "top_expert_indices": top_indices.tolist(),
            "top_expert_scores": top_scores.tolist(),
            "total_importance": float(top_scores.sum()),
            "importance_fraction": float(top_scores.sum() / importance.sum()),
        }

        print(f"  Layer {layer_idx:2d}: Experts {top_indices.tolist()} "
              f"(coverage: {selection[layer_idx]['importance_fraction']:.1%})")

    return selection


def generate_diagnostic_plots(
    layer_results: List[Dict],
    selection: Dict,
    distribution_stats: Dict,
    plot_dir: Path,
):
    """
    Generate diagnostic plots for PCA analysis.

    Args:
        layer_results: List of per-layer PCA results
        selection: Dictionary with top-K expert selections
        plot_dir: Directory to save plots
    """
    print(f"\nGenerating diagnostic plots in {plot_dir}...")
    plot_dir.mkdir(parents=True, exist_ok=True)

    num_layers = len(layer_results)

    # Plot 1: Variance explained by top components (per layer)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Subplot 1: Variance explained by first N components
    n_show = 20
    variance_data = []
    for result in layer_results:
        cumulative_var = np.cumsum(result["explained_variance_ratio"])
        variance_data.append(cumulative_var[:n_show])

    variance_array = np.array(variance_data).T
    im = axes[0].imshow(variance_array, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Principal Component')
    axes[0].set_title('Cumulative Variance Explained by Top-20 Components')
    plt.colorbar(im, ax=axes[0], label='Cumulative Variance')

    # Subplot 2: Number of components for 90% variance
    components_90 = []
    for result in layer_results:
        cumulative_var = np.cumsum(result["explained_variance_ratio"])
        n_90 = np.argmax(cumulative_var >= 0.90) + 1 if cumulative_var[-1] >= 0.90 else len(cumulative_var)
        components_90.append(n_90)

    axes[1].bar(range(num_layers), components_90, color='steelblue')
    axes[1].axhline(y=np.mean(components_90), color='red', linestyle='--',
                    label=f'Mean: {np.mean(components_90):.1f}')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Number of Components')
    axes[1].set_title('Components Required for 90% Variance')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / 'variance_explained.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: variance_explained.png")

    # Plot 2: Expert importance heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    importance_matrix = np.array([r["expert_importance"] for r in layer_results])
    im = ax.imshow(importance_matrix.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Expert Index')
    ax.set_title('Expert Importance Scores (PCA-based)')
    plt.colorbar(im, ax=ax, label='Importance Score')

    # Mark selected experts
    for layer_idx, info in selection.items():
        for expert_idx in info["top_expert_indices"]:
            ax.plot(layer_idx, expert_idx, 'b*', markersize=8, markeredgecolor='cyan', markeredgewidth=1)

    plt.tight_layout()
    plt.savefig(plot_dir / 'expert_importance_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: expert_importance_heatmap.png")

    # Plot 3: Per-layer variance distribution
    # Dynamic grid based on number of layers
    num_layers = len(layer_results)
    if num_layers == 24:
        # 20B: 4x6 grid
        grid_rows, grid_cols = 4, 6
    elif num_layers == 36:
        # 120B: 6x6 grid
        grid_rows, grid_cols = 6, 6
    else:
        # Custom: auto-calculate grid
        grid_cols = 6
        grid_rows = (num_layers + grid_cols - 1) // grid_cols

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(18, grid_rows * 3))
    axes = axes.flatten()

    for idx, result in enumerate(layer_results):  # Show all layers
        ax = axes[idx]
        variance = result["explained_variance_ratio"][:10]  # Top 10 components
        ax.bar(range(len(variance)), variance, color='steelblue')
        ax.set_title(f'Layer {idx}', fontsize=8)
        ax.set_ylim(0, 1)
        if idx % grid_cols == 0:
            ax.set_ylabel('Variance', fontsize=7)
        if idx >= (grid_rows - 1) * grid_cols:
            ax.set_xlabel('PC', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots for non-rectangular grids
    for idx in range(num_layers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Variance Explained by Top-10 Components per Layer', fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_dir / 'per_layer_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_layer_variance.png")

    # Plot 4: Top expert selection visualization
    fig, ax = plt.subplots(figsize=(12, 8))

    selected_experts = []
    for layer_idx in range(num_layers):
        experts = selection[layer_idx]["top_expert_indices"]
        selected_experts.extend([(layer_idx, exp) for exp in experts])

    layers, experts = zip(*selected_experts)
    ax.scatter(layers, experts, c='red', s=50, alpha=0.6, edgecolors='darkred')
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Expert Index')
    ax.set_title('Selected Top-4 Experts per Layer')
    ax.grid(alpha=0.3)
    ax.set_xlim(-1, num_layers)
    ax.set_ylim(-5, 133)

    plt.tight_layout()
    plt.savefig(plot_dir / 'selected_experts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: selected_experts.png")

    # Plot 5: Expert importance distribution (all 128 experts)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Distribution for a sample of layers
    ax = axes[0, 0]
    sample_layers = [0, 11, 23, 35]  # First, early-mid, late-mid, last
    for layer_idx in sample_layers:
        if layer_idx < num_layers:
            importance = layer_results[layer_idx]["expert_importance"]
            sorted_importance = np.sort(importance)[::-1]
            ax.plot(range(128), sorted_importance, label=f'Layer {layer_idx}', alpha=0.7)

    ax.set_xlabel('Expert Rank')
    ax.set_ylabel('Importance Score')
    ax.set_title('Expert Importance Distribution (Sample Layers)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')

    # Subplot 2: Cumulative importance
    ax = axes[0, 1]
    avg_cumulative = np.zeros(128)
    for result in layer_results:
        importance = result["expert_importance"]
        sorted_importance = np.sort(importance)[::-1]
        cumulative = np.cumsum(sorted_importance) / sorted_importance.sum()
        avg_cumulative += cumulative
    avg_cumulative /= len(layer_results)

    ax.plot(range(128), avg_cumulative * 100, linewidth=2, color='steelblue')
    ax.axhline(y=50, color='red', linestyle='--', label='50% importance')
    ax.axhline(y=75, color='orange', linestyle='--', label='75% importance')
    ax.axhline(y=90, color='green', linestyle='--', label='90% importance')
    ax.axvline(x=4, color='purple', linestyle=':', label='Top-4 (selected)', linewidth=2)
    ax.set_xlabel('Number of Top Experts')
    ax.set_ylabel('Cumulative Importance (%)')
    ax.set_title('Average Cumulative Importance Across Layers')
    ax.legend()
    ax.grid(alpha=0.3)

    # Subplot 3: Top-K coverage heatmap
    ax = axes[1, 0]
    coverage_data = []
    # Get layer keys in order (layer_0, layer_1, ...)
    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        layer_stats = distribution_stats['layers'][layer_key]
        coverage_data.append([
            layer_stats['top_4_fraction'] * 100,
            layer_stats['top_8_fraction'] * 100,
            layer_stats['top_16_fraction'] * 100,
            layer_stats['top_32_fraction'] * 100,
        ])

    coverage_array = np.array(coverage_data)
    im = ax.imshow(coverage_array.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Top-4', 'Top-8', 'Top-16', 'Top-32'])
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Expert Group')
    ax.set_title('Importance Coverage by Top-K Experts (%)')
    plt.colorbar(im, ax=ax, label='% of Total Importance')

    # Add text annotations
    for i in range(4):
        for j in range(num_layers):
            text = ax.text(j, i, f'{coverage_array[j, i]:.0f}',
                          ha="center", va="center", color="black", fontsize=6)

    # Subplot 4: Distribution pattern summary
    ax = axes[1, 1]
    # Show average importance decay
    avg_importance = np.zeros(128)
    for result in layer_results:
        importance = result["expert_importance"]
        sorted_importance = np.sort(importance)[::-1]
        avg_importance += sorted_importance
    avg_importance /= len(layer_results)

    # Normalize to [0, 1]
    avg_importance_norm = avg_importance / avg_importance.max()

    ax.bar(range(128), avg_importance_norm, color='steelblue', alpha=0.7)
    ax.axvline(x=4, color='red', linestyle='--', linewidth=2, label='Top-4 cutoff')
    ax.set_xlabel('Expert Rank')
    ax.set_ylabel('Normalized Importance')
    ax.set_title('Average Expert Importance Distribution (Normalized)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Highlight different regions
    ax.axvspan(0, 4, alpha=0.2, color='green', label='Top-4')
    ax.axvspan(4, 16, alpha=0.1, color='yellow')
    ax.axvspan(16, 64, alpha=0.05, color='orange')

    plt.tight_layout()
    plt.savefig(plot_dir / 'expert_distribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: expert_distribution_analysis.png")


def save_pca_stats(
    selection: Dict,
    metadata: Dict,
    output_path: Path,
    top_k: int = 4,
):
    """
    Save PCA statistics in format compatible with convert_add_shared_expert.py.

    Args:
        selection: Dictionary with top-K expert selections
        metadata: Original metadata from router probability collection
        output_path: Output JSON file path
        top_k: Number of experts selected (for method name)
    """
    stats = {
        "method": f"pca_top{top_k}",
        "num_layers": len(selection),
        "num_experts": metadata.get("num_experts", 128),
        "experts_per_layer": {k: v for k, v in metadata.items() if k != "router_probs"},
        "layers": {},
    }

    for layer_idx, info in selection.items():
        stats["layers"][f"layer_{layer_idx}"] = {
            "top_expert_indices": info["top_expert_indices"],
            "top_expert_scores": info["top_expert_scores"],
            "importance_fraction": info["importance_fraction"],
        }

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved PCA stats to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Perform PCA analysis on router probabilities"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input NPZ file with router probabilities",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pca_stats.json",
        help="Output JSON file for PCA statistics",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=24,
        help="Number of top experts to select per layer (default: 24, recommended for even distributions)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=20,
        help="Number of principal components to compute",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots/pca",
        help="Directory for diagnostic plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    args = parser.parse_args()

    # Create output directories
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_dir = Path(args.plot_dir)
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PCA-Based Expert Selection Analysis")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Top-K: {args.top_k}")
    print(f"PCA components: {args.n_components}")
    print("=" * 80)

    # Load router probabilities
    print("\nLoading router probabilities...")
    data = np.load(args.input)
    router_probs = data["router_probs"]
    metadata = json.loads(str(data["metadata"]))

    print(f"Loaded data shape: {router_probs.shape}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}")

    # Perform PCA analysis
    layer_results = perform_pca_analysis(
        router_probs=router_probs,
        n_components=args.n_components,
    )

    # Analyze expert importance distribution
    distribution_stats = analyze_expert_distribution(
        layer_results=layer_results,
    )

    # Select top-K experts per layer
    selection = select_top_k_experts(
        layer_results=layer_results,
        top_k=args.top_k,
    )

    # Generate diagnostic plots
    if not args.no_plots:
        generate_diagnostic_plots(
            layer_results=layer_results,
            selection=selection,
            distribution_stats=distribution_stats,
            plot_dir=plot_dir,
        )

    # Save results
    save_pca_stats(
        selection=selection,
        metadata=metadata,
        output_path=output_path,
        top_k=args.top_k,
    )

    # Save distribution analysis
    distribution_output = output_path.with_name(output_path.stem + '_distribution.json')
    with open(distribution_output, 'w') as f:
        json.dump(distribution_stats, f, indent=2)
    print(f"\nSaved distribution analysis to {distribution_output}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Analysis Summary")
    print("=" * 80)

    # Global statistics
    all_importance_fractions = [info["importance_fraction"] for info in selection.values()]
    avg_importance = np.mean(all_importance_fractions)
    min_importance = np.min(all_importance_fractions)
    max_importance = np.max(all_importance_fractions)

    print(f"Average importance coverage: {avg_importance:.1%}")
    print(f"Min/Max coverage: {min_importance:.1%} / {max_importance:.1%}")

    # Expert reuse statistics
    all_selected_experts = []
    for info in selection.values():
        all_selected_experts.extend(info["top_expert_indices"])

    unique_experts = len(set(all_selected_experts))
    total_selections = len(all_selected_experts)

    print(f"\nExpert reuse: {unique_experts} unique experts across {total_selections} selections")
    print(f"Reuse factor: {total_selections / unique_experts:.2f}x")

    # Distribution pattern analysis
    print("\n" + "=" * 80)
    print("Expert Distribution Pattern Analysis")
    print("=" * 80)

    # Get number of layers
    num_layers_dist = len(distribution_stats['layers'])

    # Average across all layers (using string keys layer_0, layer_1, ...)
    avg_top_4 = np.mean([distribution_stats['layers'][f'layer_{i}']['top_4_fraction'] for i in range(num_layers_dist)])
    avg_top_8 = np.mean([distribution_stats['layers'][f'layer_{i}']['top_8_fraction'] for i in range(num_layers_dist)])
    avg_top_16 = np.mean([distribution_stats['layers'][f'layer_{i}']['top_16_fraction'] for i in range(num_layers_dist)])
    avg_top_32 = np.mean([distribution_stats['layers'][f'layer_{i}']['top_32_fraction'] for i in range(num_layers_dist)])

    print(f"\nAverage importance captured by top-K experts:")
    print(f"  Top-4:  {avg_top_4:.1%}")
    print(f"  Top-8:  {avg_top_8:.1%}")
    print(f"  Top-16: {avg_top_16:.1%}")
    print(f"  Top-32: {avg_top_32:.1%}")

    # Determine distribution pattern
    print(f"\nDistribution pattern:")
    if avg_top_4 > 0.7:
        print(f"  ⚠️  STEEP CLIFF - Top-4 experts dominate ({avg_top_4:.1%} importance)")
        print(f"      → Strong concentration, top-4 selection is well-justified")
    elif avg_top_4 > 0.5:
        print(f"  📊 MODERATE CONCENTRATION - Top-4 capture {avg_top_4:.1%}")
        print(f"      → Reasonable selection, but some importance in tail")
    elif avg_top_4 > 0.35:
        print(f"  📈 GRADUAL DECLINE - Top-4 only capture {avg_top_4:.1%}")
        print(f"      → Consider selecting more experts (top-8 or top-16)")
    else:
        print(f"  ⚖️  EVEN DISTRIBUTION - Top-4 only {avg_top_4:.1%}")
        print(f"      → High expert diversity, many experts contribute equally")

    # Check for cliff locations
    cliff_positions = []
    for i in range(num_layers_dist):
        max_gap_idx = distribution_stats['layers'][f'layer_{i}']['max_gap_idx']
        cliff_positions.append(max_gap_idx)

    avg_cliff_position = np.mean(cliff_positions)
    print(f"\nAverage largest gap position: rank {avg_cliff_position:.1f}")
    if avg_cliff_position < 10:
        print(f"  → Sharp drop-off early (within top-10)")
    elif avg_cliff_position < 32:
        print(f"  → Moderate drop-off (within top-32)")
    else:
        print(f"  → Gradual decline (beyond top-32)")

    print("\nAnalysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
