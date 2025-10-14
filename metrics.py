import numpy as np

from collections import Counter, defaultdict
from itertools import combinations
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
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
    raise NotImplemented()

def calculate_npmi(gmm, corpus, keyed_vectors, top_n=10, window_size=10):
    """
    Calculates Normalized Pointwise Mutual Information (NPMI) for word clusters.

    Args:
        gmm: GaussianMixture model fitted on embeddings.
        corpus: List of documents (each as string).
        keyed_vectors: Word2Vec KeyedVectors or model.wv.
        top_n: Number of top words per cluster.
        window_size: Window size for cooccurrence.

    Returns:
        float: Mean NPMI across all clusters.
    """

    def get_word_cooccurrence_counts(corpus_tokens, window_size=5):
        cooccurrence = defaultdict(int)
        word_counts = Counter(corpus_tokens)
        total_windows = 0

        for i in range(len(corpus_tokens)):
            window = corpus_tokens[i + 1:i + 1 + window_size]
            for w1 in [corpus_tokens[i]]:
                for w2 in window:
                    if w1 != w2:
                        pair = tuple(sorted((w1, w2)))
                        cooccurrence[pair] += 1
            total_windows += 1

        return cooccurrence, word_counts, total_windows

    def calculate_npmi_from_counts(word_list, cooccurrence, word_counts, total_windows, total_word_count):
        if len(word_list) < 2:
            return 0  # Can't calculate NPMI for < 2 words

        npmi_scores = []
        for w1, w2 in combinations(word_list, 2):
            pair = tuple(sorted((w1, w2)))
            count_joint = cooccurrence.get(pair, 0)

            # Add smoothing to avoid zero counts
            count_joint += 1e-10

            # Use different probability calculation
            p_joint = count_joint / (total_windows + 1e-10)
            p1 = (word_counts[w1] + 1e-10) / (total_word_count + 1e-10)
            p2 = (word_counts[w2] + 1e-10) / (total_word_count + 1e-10)

            if p_joint > 0:
                pmi = np.log(p_joint / (p1 * p2))
                # Use standard NPMI normalization
                npmi = pmi / (-np.log(p_joint)) if p_joint < 1 else 0
                npmi_scores.append(npmi)

        return np.mean(npmi_scores) if npmi_scores else 0

    from sklearn.metrics.pairwise import cosine_similarity

    # Step 1: Tokenize corpus
    corpus_tokens = [token for doc in corpus for token in word_tokenize(doc.lower())]

    if not corpus_tokens:
        return 0.0

    # Step 2: Cooccurrence counts
    cooccurrence, word_counts, total_windows = get_word_cooccurrence_counts(corpus_tokens, window_size)
    total_word_count = sum(word_counts.values())

    # Step 3: Get top words per cluster
    from itertools import islice

    def get_top_n_nearest_points(gmm, keyed_vectors, corpus_tokens, top_n):
        cluster_centers = gmm.means_
        vocab = list(set(corpus_tokens))
        embeddings = []
        valid_words = []

        for word in vocab:
            if word in keyed_vectors and keyed_vectors.has_index_for(word):
                embeddings.append(keyed_vectors[word])
                valid_words.append(word)

        embeddings = np.array(embeddings)
        labels = gmm.predict(embeddings)
        top_words_per_cluster = []

        for k in range(gmm.n_components):
            indices = np.where(labels == k)[0]
            cluster_words = [valid_words[i] for i in indices]
            cluster_vectors = embeddings[indices]
            if len(cluster_vectors) == 0:
                top_words_per_cluster.append([])
                continue
            centroid = cluster_centers[k]
            sims = cosine_similarity(cluster_vectors, centroid.reshape(1, -1)).flatten()
            top_indices = np.argsort(sims)[-top_n:]
            top_words = [cluster_words[i] for i in top_indices]
            top_words_per_cluster.append(top_words)

        return top_words_per_cluster

    top_words_per_cluster = get_top_n_nearest_points(gmm, keyed_vectors, corpus_tokens, top_n)

    # Step 4: Compute NPMI per cluster
    npmi_scores = []
    for word_list in top_words_per_cluster:
        score = calculate_npmi_from_counts(word_list, cooccurrence, word_counts, total_windows, total_word_count)
        npmi_scores.append(score)

    return np.mean(npmi_scores) if npmi_scores else 0.0
