import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings


def calculate_scci_improved(embedding_corpus, labels, epsilon=1e-10,
                            weight_by_size=False, handle_singletons='exclude',
                            normalize_output=False):
    """
    Calculates the Improved Semantic Cohesive Clustering Index (SCCI).

    Args:
        embedding_corpus (np.ndarray): Embeddings of the words (shape: [n_samples, n_features]).
        labels (np.ndarray): Cluster labels for each word (shape: [n_samples]).
        epsilon (float): Small constant to avoid division by zero.
        weight_by_size (bool): Whether to weight ST by cluster sizes.
        handle_singletons (str): How to handle single-point clusters:
            - 'exclude': Don't include in calculations
            - 'perfect': Treat as having perfect tightness (ST=1)
            - 'zero': Treat as having no tightness (ST=0)
        normalize_output (bool): Whether to normalize SCCI to [0, 1] range.

    Returns:
        dict: Dictionary containing:
            - 'scci': SCCI score
            - 'st': Semantic Tightness
            - 'so': Semantic Overlap
            - 'n_clusters': Number of clusters
            - 'n_singletons': Number of singleton clusters
            - 'interpretation': String interpretation of the score
    """
    if len(embedding_corpus) == 0 or len(labels) == 0:
        return {'scci': 0.0, 'st': 0.0, 'so': 0.0, 'n_clusters': 0,
                'n_singletons': 0, 'interpretation': 'No data'}

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Edge case: single cluster
    if n_clusters == 1:
        warnings.warn("Only one cluster found. SCCI undefined (no inter-cluster comparison possible).")
        return {'scci': float('inf'), 'st': 1.0, 'so': 0.0, 'n_clusters': 1,
                'n_singletons': 0, 'interpretation': 'Single cluster - perfect separation, undefined SCCI'}

    # 1. Compute Semantic Tightness (ST)
    tightness_scores = []
    cluster_sizes = []
    n_singletons = 0
    valid_clusters_for_st = []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_embeddings = embedding_corpus[indices]
        n_k = len(cluster_embeddings)

        if n_k == 1:
            n_singletons += 1
            if handle_singletons == 'perfect':
                tightness_scores.append(1.0)
                cluster_sizes.append(1)
                valid_clusters_for_st.append(label)
            elif handle_singletons == 'zero':
                tightness_scores.append(0.0)
                cluster_sizes.append(1)
                valid_clusters_for_st.append(label)
            # If 'exclude', we don't add anything
        elif n_k >= 2:
            valid_clusters_for_st.append(label)
            similarities = cosine_similarity(cluster_embeddings)
            upper_triangle_indices = np.triu_indices_from(similarities, k=1)
            cluster_similarities = similarities[upper_triangle_indices]
            st_k = np.mean(cluster_similarities)
            tightness_scores.append(st_k)
            cluster_sizes.append(n_k)

    # Calculate mean tightness
    if tightness_scores:
        if weight_by_size:
            weights = np.array(cluster_sizes) / np.sum(cluster_sizes)
            mean_tightness = np.average(tightness_scores, weights=weights)
        else:
            mean_tightness = np.mean(tightness_scores)
    else:
        mean_tightness = 0.0
        warnings.warn("No valid clusters for ST calculation (all singletons with 'exclude' option).")

    # 2. Compute Semantic Overlap (SO)
    centroids = []
    valid_clusters_for_so = []

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_embeddings = embedding_corpus[indices]

        # For SO, we include all clusters (even singletons)
        if handle_singletons == 'exclude' and len(cluster_embeddings) == 1:
            continue

        centroid = np.mean(cluster_embeddings, axis=0)
        centroids.append(centroid)
        valid_clusters_for_so.append(label)

    if len(centroids) < 2:
        warnings.warn("Less than 2 valid clusters for SO calculation.")
        mean_overlap = 0.0
    else:
        centroids = np.stack(centroids)
        centroid_similarities = cosine_similarity(centroids)
        upper_triangle_indices = np.triu_indices_from(centroid_similarities, k=1)
        centroid_overlap = centroid_similarities[upper_triangle_indices]
        # Apply max(0, .) to each similarity, then average ALL pairs
        mean_overlap = np.mean(np.maximum(0, centroid_overlap))

    # 3. Calculate SCCI
    scci_score = mean_tightness / (mean_overlap + epsilon)

    # 4. Normalize if requested
    if normalize_output:
        # Using sigmoid-like transformation to bound the score
        scci_normalized = scci_score / (1 + scci_score)
        scci_score = scci_normalized

    # 5. Provide interpretation
    if scci_score > 10:
        interpretation = "Excellent clustering (very tight and well-separated)"
    elif scci_score > 5:
        interpretation = "Good clustering (tight and separated)"
    elif scci_score > 2:
        interpretation = "Moderate clustering (some tightness and separation)"
    elif scci_score > 1:
        interpretation = "Poor clustering (loose or overlapping)"
    else:
        interpretation = "Very poor clustering (very loose or highly overlapping)"

    if normalize_output:
        # Adjusted interpretation for normalized scores
        if scci_score > 0.9:
            interpretation = "Excellent clustering"
        elif scci_score > 0.8:
            interpretation = "Good clustering"
        elif scci_score > 0.6:
            interpretation = "Moderate clustering"
        elif scci_score > 0.4:
            interpretation = "Poor clustering"
        else:
            interpretation = "Very poor clustering"

    return {
        'scci': round(scci_score, 4),
        'st': round(mean_tightness, 4),
        'so': round(mean_overlap, 4),
        'n_clusters': n_clusters,
        'n_singletons': n_singletons,
        'interpretation': interpretation
    }


def calculate_scci_gmm(embedding_corpus, gmm_model, epsilon=1e-10, normalize_output=True):
    """
    Calculates SCCI for Gaussian Mixture Models.

    Args:
        embedding_corpus (np.ndarray): Embeddings [n_samples, n_features]
        gmm_model: Fitted sklearn.mixture.GaussianMixture object
        epsilon (float): Small constant to avoid division by zero

    Returns:
        dict: GMM-SCCI results
    """
    from scipy.spatial.distance import mahalanobis

    n_components = gmm_model.n_components
    weights = gmm_model.weights_  # π_k
    means = gmm_model.means_  # μ_k
    covariances = gmm_model.covariances_  # Σ_k

    # Get soft assignments
    responsibilities = gmm_model.predict_proba(embedding_corpus)  # γ_ik

    # 1. Compute Probabilistic Semantic Tightness (PST)
    pst_scores = []

    for k in range(n_components):
        gamma_k = responsibilities[:, k]

        # Compute weighted pairwise similarities
        numerator = 0.0
        denominator = 0.0

        for i in range(len(embedding_corpus)):
            for j in range(i + 1, len(embedding_corpus)):
                weight = gamma_k[i] * gamma_k[j]
                if weight > 1e-6:  # Skip negligible contributions
                    sim = np.dot(embedding_corpus[i], embedding_corpus[j]) / (
                            np.linalg.norm(embedding_corpus[i]) * np.linalg.norm(embedding_corpus[j])
                    )
                    numerator += weight * sim
                    denominator += weight

        if denominator > 0:
            pst_k = numerator / denominator
            pst_scores.append(pst_k)
        else:
            pst_scores.append(0.0)

    # Weight by mixing coefficients
    pst = np.sum(weights * np.array(pst_scores))

    # 2. Compute Probabilistic Semantic Overlap (PSO)
    pso = 0.0

    for l in range(n_components):
        for m in range(l + 1, n_components):
            # Bhattacharyya coefficient for multivariate Gaussians
            mu_diff = means[l] - means[m]
            if gmm_model.covariance_type == 'full':
                sigma_avg = (covariances[l] + covariances[m]) / 2
                sigma_l = covariances[l]
                sigma_m = covariances[m]
            elif gmm_model.covariance_type == 'diag':
                sigma_avg = np.diag((covariances[l] + covariances[m]) / 2)
                sigma_l = np.diag(covariances[l])
                sigma_m = np.diag(covariances[m])
            elif gmm_model.covariance_type == 'spherical':
                sigma_avg = np.eye(len(mu_diff)) * (covariances[l] + covariances[m]) / 2
                sigma_l = np.eye(len(mu_diff)) * covariances[l]
                sigma_m = np.eye(len(mu_diff)) * covariances[m]
            else:  # tied
                sigma_avg = covariances
                sigma_l = covariances
                sigma_m = covariances
            # Mahalanobis term
            try:
                sigma_avg_inv = np.linalg.inv(sigma_avg)
                mahal_dist = mu_diff.T @ sigma_avg_inv @ mu_diff
                # Determinant term
                det_avg = np.linalg.det(sigma_avg)
                det_l = np.linalg.det(sigma_l)
                det_m = np.linalg.det(sigma_m)
                det_term = det_avg / np.sqrt(det_l * det_m)
                bc_lm = np.exp(-0.125 * mahal_dist) * np.sqrt(det_term)
                # Weight by component probabilities
                pso += weights[l] * weights[m] * bc_lm

            except np.linalg.LinAlgError:
                # Singular covariance, skip this pair
                continue

    # 3. Calculate GMM-SCCI
    gmm_scci = pst / (pso + epsilon)

    if normalize_output:
        # Using sigmoid-like transformation to bound the score
        scci_normalized = gmm_scci / (1 + gmm_scci)
        gmm_scci = scci_normalized

    return {
        'gmm_scci': round(gmm_scci, 4),
        'pst': round(pst, 4),
        'pso': round(pso, 4),
        'n_components': n_components
    }

# Additional utility function for robustness testing
def calculate_scci_bootstrap(embedding_corpus, labels, n_bootstrap=100,
                             sample_fraction=0.8, **scci_kwargs):
    """
    Calculate SCCI with bootstrap confidence intervals.

    Args:
        embedding_corpus: Word embeddings
        labels: Cluster labels
        n_bootstrap: Number of bootstrap samples
        sample_fraction: Fraction of data to sample in each iteration
        **scci_kwargs: Additional arguments for calculate_scci_improved

    Returns:
        dict: SCCI statistics with confidence intervals
    """
    n_samples = len(embedding_corpus)
    scci_scores = []

    for _ in range(n_bootstrap):
        # Sample indices
        sample_size = int(n_samples * sample_fraction)
        sample_indices = np.random.choice(n_samples, sample_size, replace=True)

        # Get sampled data
        sampled_embeddings = embedding_corpus[sample_indices]
        sampled_labels = labels[sample_indices]

        # Calculate SCCI
        result = calculate_scci_improved(sampled_embeddings, sampled_labels, **scci_kwargs)
        scci_scores.append(result['scci'])

    scci_scores = np.array(scci_scores)

    return {
        'mean_scci': np.mean(scci_scores),
        'std_scci': np.std(scci_scores),
        'ci_lower': np.percentile(scci_scores, 2.5),
        'ci_upper': np.percentile(scci_scores, 97.5),
        'median_scci': np.median(scci_scores)
    }


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)

    # Create 3 well-separated clusters
    cluster1 = np.random.randn(20, 10) + np.array([5] * 10)
    cluster2 = np.random.randn(15, 10) + np.array([-5] * 10)
    cluster3 = np.random.randn(25, 10) + np.array([0, 0, 0, 0, 0, 10, 10, 10, 10, 10])

    embeddings = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0] * 20 + [1] * 15 + [2] * 25)

    # Test different configurations
    print("Standard SCCI:")
    result = calculate_scci_improved(embeddings, labels)
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\nWeighted by size:")
    result = calculate_scci_improved(embeddings, labels, weight_by_size=True)
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\nNormalized output:")
    result = calculate_scci_improved(embeddings, labels, normalize_output=True)
    for key, value in result.items():
        print(f"  {key}: {value}")

    print("\nBootstrap confidence intervals:")
    bootstrap_result = calculate_scci_bootstrap(embeddings, labels, n_bootstrap=100)
    for key, value in bootstrap_result.items():
        print(f"  {key}: {value:.4f}")