from itertools import combinations

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

from utils import get_top_n_nearest_points


def calculate_semantic_coherence(w2v_model, gmm, corpus_words, top_n=10):
    # Obtener vectores según el tipo de modelo
    if hasattr(w2v_model, 'wv'):
        keyed_vectors = w2v_model.wv
    else:
        keyed_vectors = w2v_model

    # Obtener los top_n puntos por cluster una sola vez
    top_words_per_cluster = get_top_n_nearest_points(gmm, keyed_vectors, corpus_words, top_n)

    coherence_scores = []
    for cluster_words in top_words_per_cluster:
        similarities = []
        for word1, word2 in combinations(cluster_words, 2):
            if word1 in keyed_vectors and word2 in keyed_vectors:
                try:
                    sim = cosine_similarity([keyed_vectors[word1]], [keyed_vectors[word2]])[0][0]
                    similarities.append(sim)
                except:
                    continue
        # Media de similitudes del cluster (0 si está vacío)
        coherence_scores.append(np.mean(similarities) if similarities else 0)

    # Media global de todas las medias de clusters
    return np.mean(coherence_scores) if coherence_scores else 0


def calculate_perplexity(gmm, embedding_corpus):
    # Perplexity is a commonly used metric for language models; for GMMs, it can be adapted
    # as the exponential of the average negative log-likelihood.
    log_likelihood = np.sum(gmm.score_samples(embedding_corpus))
    return np.exp(-log_likelihood / len(embedding_corpus))


def calculate_topic_diversity(w2v_model, gmm, corpus, top_n_words):
    unique_words = set()
    total_words = 0
    for idx in range(len(gmm.means_)):
        cluster_words = get_top_n_nearest_points(gmm, w2v_model, corpus, top_n_words)[
            idx
        ]
        unique_words.update(cluster_words)
        total_words += len(cluster_words)
    return len(unique_words) / total_words if total_words > 0 else 0


def calculate_intra_distances(gmm, embedding_corpus):
    intra_distances = []
    for i, mean1 in enumerate(gmm.means_):
        intra_distances.append(np.mean(pairwise_distances(embedding_corpus, [mean1])))

    return np.mean(intra_distances)


def calculate_inter_distances(gmm, embedding_corpus):
    inter_distances = []
    for i, mean1 in enumerate(gmm.means_):
        for j, mean2 in enumerate(gmm.means_):
            if i < j:
                inter_distances.append(np.linalg.norm(mean1 - mean2))
    return np.mean(inter_distances)


def calculate_npmi(gmm, corpus_words, w2v_model, top_n=10):
    # Obtener modelo base
    if hasattr(w2v_model, 'wv'):
        keyed_vectors = w2v_model.wv
    else:
        keyed_vectors = w2v_model

    # Total de ocurrencias (frecuencia total)
    total_word_count = sum(
        keyed_vectors.get_vecattr(word, "count")
        for word in corpus_words
        if word in keyed_vectors and keyed_vectors.has_index_for(word)
    )

    # Early exit si no hay conteo válido
    if total_word_count == 0:
        return 0

    # Obtener palabras más cercanas por cluster
    top_words_per_cluster = get_top_n_nearest_points(gmm, keyed_vectors, corpus_words, top_n)

    npmi_scores = []
    for cluster_words in top_words_per_cluster:
        pair_scores = []
        for word1, word2 in combinations(cluster_words, 2):
            if all(word in keyed_vectors and keyed_vectors.has_index_for(word)
                   for word in [word1, word2]):
                try:
                    count1 = keyed_vectors.get_vecattr(word1, "count")
                    count2 = keyed_vectors.get_vecattr(word2, "count")
                    p1 = count1 / total_word_count
                    p2 = count2 / total_word_count

                    # Proxy para p(word1 ∩ word2)
                    p_joint = keyed_vectors.similarity(word1, word2)

                    if p1 > 0 and p2 > 0 and p_joint > 0:
                        npmi = np.log(p_joint / (p1 * p2)) / -np.log(p_joint)
                        pair_scores.append(npmi)
                except:
                    continue
        npmi_scores.append(np.mean(pair_scores) if pair_scores else 0)

    return np.mean(npmi_scores) if npmi_scores else 0


def calculate_scci(embedding_corpus, labels, epsilon=1e-10):
    """
    Calculates the Semantic Cohesive Clustering Index (SCCI).

    Args:
        embedding_corpus (np.ndarray): Embeddings of the words (shape: [n_samples, n_features]).
        labels (np.ndarray): Cluster labels for each word (shape: [n_samples]).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: SCCI score.
    """
    if len(embedding_corpus) == 0 or len(labels) == 0:
        return 0.0

    unique_labels = np.unique(labels)

    # 1. Compute Semantic Tightness (ST)
    tightness_scores = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_embeddings = embedding_corpus[indices]

        if len(cluster_embeddings) < 2:
            # Skip clusters with less than 2 points
            continue

        similarities = cosine_similarity(cluster_embeddings)
        upper_triangle_indices = np.triu_indices_from(similarities, k=1)
        cluster_similarities = similarities[upper_triangle_indices]
        st_k = np.mean(cluster_similarities)
        tightness_scores.append(st_k)

    mean_tightness = np.mean(tightness_scores) if tightness_scores else 0.0

    # 2. Compute Semantic Overlap (SO)
    centroids = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        cluster_embeddings = embedding_corpus[indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        centroids.append(centroid)

    centroids = np.stack(centroids)
    centroid_similarities = cosine_similarity(centroids)
    upper_triangle_indices = np.triu_indices_from(centroid_similarities, k=1)
    centroid_overlap = centroid_similarities[upper_triangle_indices]

    # Only positive similarities count as overlap
    positive_overlap = centroid_overlap[centroid_overlap > 0]
    mean_overlap = np.mean(positive_overlap) if positive_overlap.size > 0 else 0.0

    # 3. Calculate SCCI
    scci_score = mean_tightness / (mean_overlap + epsilon)

    return scci_score


