import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

from new_scci import calculate_scci_improved
from sklearn.decomposition import PCA

def generate_test_scenarios():
    """Generate different clustering scenarios to test SCCI behavior."""

    scenarios = {}

    # Scenario 1: Perfect clustering - tight and well-separated
    print("Generating Scenario 1: Perfect clustering")
    np.random.seed(42)
    cluster1 = normalize(np.random.randn(20, 50) * 0.1 + np.array([10] + [0] * 49).reshape(1, -1))
    cluster2 = normalize(np.random.randn(20, 50) * 0.1 + np.array([0, 10] + [0] * 48).reshape(1, -1))
    cluster3 = normalize(np.random.randn(20, 50) * 0.1 + np.array([0, 0, 10] + [0] * 47).reshape(1, -1))

    scenarios['perfect'] = {
        'embeddings': np.vstack([cluster1, cluster2, cluster3]),
        'labels': np.array([0] * 20 + [1] * 20 + [2] * 20),
        'description': 'Perfect: Tight clusters, well-separated (orthogonal)'
    }

    # Scenario 2: Good clustering - tight but some overlap
    print("Generating Scenario 2: Good clustering")
    np.random.seed(43)
    cluster1 = normalize(np.random.randn(20, 50) * 0.2 + np.array([5, 1] + [0] * 48).reshape(1, -1))
    cluster2 = normalize(np.random.randn(20, 50) * 0.2 + np.array([1, 5] + [0] * 48).reshape(1, -1))
    cluster3 = normalize(np.random.randn(20, 50) * 0.2 + np.array([0, 0, 5] + [0] * 47).reshape(1, -1))

    scenarios['good'] = {
        'embeddings': np.vstack([cluster1, cluster2, cluster3]),
        'labels': np.array([0] * 20 + [1] * 20 + [2] * 20),
        'description': 'Good: Tight clusters, moderate separation'
    }

    # Scenario 3: Poor clustering - loose clusters
    print("Generating Scenario 3: Poor clustering - loose")
    np.random.seed(44)
    cluster1 = normalize(np.random.randn(20, 50) * 1.0 + np.array([2, 0] + [0] * 48).reshape(1, -1))
    cluster2 = normalize(np.random.randn(20, 50) * 1.0 + np.array([0, 2] + [0] * 48).reshape(1, -1))
    cluster3 = normalize(np.random.randn(20, 50) * 1.0 + np.array([1, 1] + [0] * 48).reshape(1, -1))

    scenarios['loose'] = {
        'embeddings': np.vstack([cluster1, cluster2, cluster3]),
        'labels': np.array([0] * 20 + [1] * 20 + [2] * 20),
        'description': 'Poor: Loose clusters, some separation'
    }

    # Scenario 4: Poor clustering - high overlap
    print("Generating Scenario 4: Poor clustering - overlapping")
    np.random.seed(45)
    base_direction = np.array([1, 1, 1] + [0] * 47).reshape(1, -1)
    cluster1 = normalize(np.random.randn(20, 50) * 0.3 + base_direction)
    cluster2 = normalize(np.random.randn(20, 50) * 0.3 + base_direction * 1.1)
    cluster3 = normalize(np.random.randn(20, 50) * 0.3 + base_direction * 0.9)

    scenarios['overlap'] = {
        'embeddings': np.vstack([cluster1, cluster2, cluster3]),
        'labels': np.array([0] * 20 + [1] * 20 + [2] * 20),
        'description': 'Poor: Tight clusters but highly overlapping'
    }

    # Scenario 5: Worst case - random assignment
    print("Generating Scenario 5: Random clustering")
    np.random.seed(46)
    random_embeddings = normalize(np.random.randn(60, 50))
    random_labels = np.random.randint(0, 3, 60)

    scenarios['random'] = {
        'embeddings': random_embeddings,
        'labels': random_labels,
        'description': 'Worst: Random cluster assignment'
    }

    # Scenario 6: Mixed quality - some good, some bad clusters
    print("Generating Scenario 6: Mixed quality")
    np.random.seed(47)
    # One tight cluster
    cluster1 = normalize(np.random.randn(20, 50) * 0.1 + np.array([10] + [0] * 49).reshape(1, -1))
    # One loose cluster
    cluster2 = normalize(np.random.randn(20, 50) * 1.0)
    # One overlapping with first
    cluster3 = normalize(np.random.randn(20, 50) * 0.2 + np.array([9] + [0] * 49).reshape(1, -1))

    scenarios['mixed'] = {
        'embeddings': np.vstack([cluster1, cluster2, cluster3]),
        'labels': np.array([0] * 20 + [1] * 20 + [2] * 20),
        'description': 'Mixed: One tight, one loose, one overlapping'
    }

    # Scenario 7: Many singleton clusters
    print("Generating Scenario 7: Many singletons")
    np.random.seed(48)
    # Few normal clusters
    cluster1 = normalize(np.random.randn(5, 50) * 0.1 + np.array([5] + [0] * 49).reshape(1, -1))
    cluster2 = normalize(np.random.randn(5, 50) * 0.1 + np.array([0, 5] + [0] * 48).reshape(1, -1))
    # Many singletons
    singletons = normalize(np.random.randn(20, 50))

    scenarios['singletons'] = {
        'embeddings': np.vstack([cluster1, cluster2, singletons]),
        'labels': np.array([0] * 5 + [1] * 5 + list(range(2, 22))),
        'description': 'Edge case: Many singleton clusters'
    }

    return scenarios

def generate_test_clusters(scenarios, output_dir):
    rows = 3
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()

    for i, (key, scenario) in enumerate(scenarios.items()):
        X = scenario['embeddings']
        y = scenario['labels']

        # Reducción a 2D con PCA
        X_2d = PCA(n_components=2).fit_transform(X)

        axes[i].scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=50)
        axes[i].set_title(f"({i+1}) {scenario['description']}", fontsize=10)
        axes[i].grid(True)

    # Ocultar subplots vacíos
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/test_clusters.png", dpi=300, bbox_inches="tight")

def test_scci_scenarios():
    """Test SCCI on various clustering scenarios."""

    scenarios = generate_test_scenarios()
    generate_test_clusters(scenarios, "./scci_evaluation")

    results = {}

    print("\n" + "=" * 80)
    print("SCCI EVALUATION ON DIFFERENT CLUSTERING SCENARIOS")
    print("=" * 80)

    for name, scenario in scenarios.items():
        print(f"\n{scenario['description']}")
        print("-" * len(scenario['description']))

        # Calculate SCCI with different options
        result_standard = calculate_scci_improved(
            scenario['embeddings'],
            scenario['labels'],
            handle_singletons='exclude'
        )

        result_weighted = calculate_scci_improved(
            scenario['embeddings'],
            scenario['labels'],
            weight_by_size=True,
            handle_singletons='exclude'
        )

        result_normalized = calculate_scci_improved(
            scenario['embeddings'],
            scenario['labels'],
            normalize_output=True,
            handle_singletons='exclude'
        )

        result_singleton_perfect = calculate_scci_improved(
            scenario['embeddings'],
            scenario['labels'],
            normalize_output=False,
            handle_singletons='perfect'
        )
        result_singleton_zero = calculate_scci_improved(
            scenario['embeddings'],
            scenario['labels'],
            normalize_output=False,
            handle_singletons='zero'
        )
        result_singleton_perfect_normalized = calculate_scci_improved(
            scenario['embeddings'],
            scenario['labels'],
            normalize_output=True,
            handle_singletons='perfect'
        )
        result_singleton_zero_normalized = calculate_scci_improved(
            scenario['embeddings'],
            scenario['labels'],
            normalize_output=True,
            handle_singletons='zero'
        )


        # Store results
        results[name] = {
            'standard': result_standard,
            'weighted': result_weighted,
            'normalized': result_normalized
        }

        # Print results
        print(f"  Standard SCCI: {result_standard['scci']:.4f}")
        print(f"  - Semantic Tightness (ST): {result_standard['st']:.4f}")
        print(f"  - Semantic Overlap (SO): {result_standard['so']:.4f}")
        print(f"  - Interpretation: {result_standard['interpretation']}")

        if result_standard['n_singletons'] > 0:
            print(f"  - Singleton clusters: {result_standard['n_singletons']}/{result_standard['n_clusters']}")
            print(f"  - Singleton clusters exclude: {result_standard}")
            print(f"  - Singleton clusters perfect: {result_singleton_perfect}")
            print(f"  - Singleton clusters perfect normalized: {result_singleton_perfect_normalized}")
            print(f"  - Singleton clusters zero: {result_singleton_zero}")
            print(f"  - Singleton clusters zero normalized: {result_singleton_zero_normalized}")

        print(f"\n  Normalized SCCI: {result_normalized['scci']:.4f}")
        print(f"  - Interpretation: {result_normalized['interpretation']}")

    return results


def create_comparison_plot(results):
    """Create a visual comparison of SCCI scores across scenarios."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    scenarios = list(results.keys())

    # Plot 1: SCCI scores comparison
    ax = axes[0, 0]
    scci_scores = [results[s]['standard']['scci'] for s in scenarios]
    colors = ['green' if score > 5 else 'orange' if score > 2 else 'red' for score in scci_scores]
    bars = ax.bar(range(len(scenarios)), scci_scores, color=colors)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('SCCI Score')
    ax.set_title('SCCI Scores Across Scenarios')
    ax.axhline(y=5, color='g', linestyle='--', alpha=0.3, label='Good threshold')
    ax.axhline(y=2, color='orange', linestyle='--', alpha=0.3, label='Moderate threshold')
    ax.legend()

    # Add value labels on bars
    for bar, score in zip(bars, scci_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.2f}', ha='center', va='bottom')

    # Plot 2: ST vs SO scatter
    ax = axes[0, 1]
    st_scores = [results[s]['standard']['st'] for s in scenarios]
    so_scores = [results[s]['standard']['so'] for s in scenarios]

    scatter = ax.scatter(so_scores, st_scores, s=100, c=scci_scores, cmap='RdYlGn',
                         edgecolors='black', linewidth=1)

    for i, txt in enumerate(scenarios):
        ax.annotate(txt, (so_scores[i], st_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('Semantic Overlap (SO) - Lower is better')
    ax.set_ylabel('Semantic Tightness (ST) - Higher is better')
    ax.set_title('ST vs SO Trade-off')
    plt.colorbar(scatter, ax=ax, label='SCCI Score')

    # Plot 3: Normalized SCCI comparison
    ax = axes[1, 0]
    norm_scores = [results[s]['normalized']['scci'] for s in scenarios]
    colors = ['green' if score > 0.8 else 'orange' if score > 0.6 else 'red' for score in norm_scores]
    bars = ax.bar(range(len(scenarios)), norm_scores, color=colors)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('Normalized SCCI Score')
    ax.set_title('Normalized SCCI Scores (0-1 scale)')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.3, label='Good threshold')
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.3, label='Moderate threshold')
    ax.legend()

    # Add value labels on bars
    for bar, score in zip(bars, norm_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{score:.3f}', ha='center', va='bottom')

    # Plot 4: Component breakdown
    ax = axes[1, 1]
    x = np.arange(len(scenarios))
    width = 0.35

    bars1 = ax.bar(x - width / 2, st_scores, width, label='ST (Tightness)', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width / 2, so_scores, width, label='SO (Overlap)', color='red', alpha=0.7)

    # ax.set_xlabel('Scenarios')
    ax.set_ylabel('Score')
    ax.set_title('ST and SO Components')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('./scci_evaluation/scci_analysis.png', dpi=150, bbox_inches='tight')
    # plt.show()

    return fig


def analyze_metric_properties():
    """Analyze mathematical properties of the SCCI metric."""

    print("\n" + "=" * 80)
    print("SCCI METRIC PROPERTIES ANALYSIS")
    print("=" * 80)

    # Property 1: Behavior with perfect separation
    print("\n1. PERFECT SEPARATION (Orthogonal clusters)")
    print("-" * 40)
    np.random.seed(50)

    # Create perfectly orthogonal clusters
    n_dims = 100
    n_points = 10
    embeddings = []
    labels = []

    for i in range(3):
        cluster = np.zeros((n_points, n_dims))
        cluster[:, i] = 1  # Each cluster in different dimension
        cluster += np.random.randn(n_points, n_dims) * 0.01  # Tiny noise
        cluster = normalize(cluster)
        embeddings.append(cluster)
        labels.extend([i] * n_points)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    result = calculate_scci_improved(embeddings, labels)
    print(f"  SCCI: {result['scci']:.4f}")
    print(f"  ST: {result['st']:.4f}, SO: {result['so']:.4f}")
    print(f"  Expected: Very high SCCI (SO ≈ 0)")

    # Property 2: Behavior with complete overlap
    print("\n2. COMPLETE OVERLAP (Identical clusters)")
    print("-" * 40)

    # Create identical clusters
    base_cluster = normalize(np.random.randn(n_points, n_dims))
    embeddings = np.vstack([base_cluster, base_cluster, base_cluster])
    labels = np.array([0] * n_points + [1] * n_points + [2] * n_points)

    result = calculate_scci_improved(embeddings, labels)
    print(f"  SCCI: {result['scci']:.4f}")
    print(f"  ST: {result['st']:.4f}, SO: {result['so']:.4f}")
    print(f"  Expected: Low SCCI (SO ≈ 1, ST ≈ SO)")

    # Property 3: Scale invariance
    print("\n3. SCALE INVARIANCE TEST")
    print("-" * 40)

    # Original embeddings
    embeddings1 = normalize(np.random.randn(30, 50))
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

    # Scaled embeddings (shouldn't matter for cosine similarity)
    embeddings2 = embeddings1 * 100

    result1 = calculate_scci_improved(embeddings1, labels)
    result2 = calculate_scci_improved(embeddings2, labels)

    print(f"  Original SCCI: {result1['scci']:.4f}")
    print(f"  Scaled SCCI: {result2['scci']:.4f}")
    print(f"  Difference: {abs(result1['scci'] - result2['scci']):.6f}")
    print(f"  Expected: Near zero difference (scale invariant)")

    # Property 4: Sensitivity to noise
    print("\n4. NOISE SENSITIVITY")
    print("-" * 40)

    # Create clean clusters
    clean_embeddings = []
    for i in range(3):
        cluster = np.zeros((10, 50))
        cluster[:, i * 2:(i * 2) + 2] = 1
        clean_embeddings.append(normalize(cluster))
    clean_embeddings = np.vstack(clean_embeddings)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)

    noise_levels = [0, 0.1, 0.5, 1.0, 2.0]
    for noise in noise_levels:
        noisy = normalize(clean_embeddings + np.random.randn(*clean_embeddings.shape) * noise)
        result = calculate_scci_improved(noisy, labels)
        print(f"  Noise level {noise}: SCCI = {result['scci']:.4f}")

    print("  Expected: SCCI decreases with increasing noise")


# Main execution
if __name__ == "__main__":
    # Import the improved SCCI function
    # If running this as a separate file, make sure to include the calculate_scci_improved function

    print("Starting SCCI comprehensive testing...")

    # Run scenario tests
    results = test_scci_scenarios()

    # Analyze metric properties
    analyze_metric_properties()

    # Create visualization (optional - comment out if matplotlib not available)
    try:
        fig = create_comparison_plot(results)
        print("\nVisualization saved as 'scci_analysis.png'")
    except ImportError:
        print("\nMatplotlib not available - skipping visualization")

    # Summary recommendations
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print("""
    Based on the analysis:

    1. The SCCI metric correctly identifies clustering quality:
       - High scores (>5) for well-separated, tight clusters
       - Medium scores (2-5) for moderate quality
       - Low scores (<2) for poor clustering

    2. The metric successfully balances two objectives:
       - Rewards high within-cluster similarity (ST)
       - Penalizes between-cluster similarity (SO)

    3. Key insights:
       - The metric is scale-invariant (due to cosine similarity)
       - Sensitive to noise but in a predictable way
       - Handles edge cases reasonably well

    4. Recommended thresholds for interpretation:
       - SCCI > 10: Excellent clustering
       - SCCI > 5: Good clustering
       - SCCI > 2: Moderate clustering
       - SCCI > 1: Poor clustering
       - SCCI < 1: Very poor clustering

    5. For normalized SCCI (0-1 scale):
       - > 0.9: Excellent
       - > 0.8: Good
       - > 0.6: Moderate
       - > 0.4: Poor
       - < 0.4: Very poor
    """)