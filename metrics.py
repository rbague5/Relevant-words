import numpy as np

from collections import Counter, defaultdict
from itertools import combinations
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from utils import get_top_n_nearest_points


def calculate_semantic_coherence(embedding_model, gmm, corpus_words, top_n=10):
    """
    Computes semantic coherence of clusters using pairwise cosine similarity of top words.
    Works for Word2Vec, FastText, GloVe, SBERT, MiniLM, E5.
    """

    # Determine embedding access
    if hasattr(embedding_model, 'wv'):
        keyed_vectors = embedding_model.wv
        def get_vector(word):
            return keyed_vectors[word]
        def has_word(word):
            return word in keyed_vectors.key_to_index
    elif hasattr(embedding_model, 'get_word_embedding'):
        # Contextual models
        def get_vector(word):
            return embedding_model.get_word_embedding(str(word))
        def has_word(word):
            return True  # contextual embeddings can embed any word
    else:
        keyed_vectors = embedding_model
        def get_vector(word):
            return keyed_vectors[word]
        def has_word(word):
            return word in getattr(keyed_vectors, 'key_to_index', set())

    # Get top N words per cluster
    top_words_per_cluster = get_top_n_nearest_points(gmm, embedding_model, corpus_words, n_points=top_n)

    coherence_scores = []

    for cluster_words in top_words_per_cluster:
        similarities = []
        for word1, word2 in combinations(cluster_words, 2):
            if has_word(word1) and has_word(word2):
                vec1 = get_vector(word1).reshape(1, -1)
                vec2 = get_vector(word2).reshape(1, -1)
                sim = cosine_similarity(vec1, vec2)[0][0]
                similarities.append(sim)
        coherence_scores.append(np.mean(similarities) if similarities else 0)

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


def calculate_npmi(gmm, embedding_model, corpus_docs, top_n=10, window_size=10):
    """
    Calculates Normalized Pointwise Mutual Information (NPMI) for word clusters.
    Works for Word2Vec / FastText / GloVe and contextual models (SBERT, MiniLM, E5).
    """

    # STATIC embeddings
    if hasattr(embedding_model, 'wv'):
        keyed_vectors = embedding_model.wv
        def has_word(word):
            return word in keyed_vectors.key_to_index
    # CONTEXTUAL embeddings
    elif hasattr(embedding_model, 'get_word_embedding'):
        def has_word(word):
            return True  # contextual embeddings can embed any word
    else:
        keyed_vectors = embedding_model
        def has_word(word):
            return word in getattr(keyed_vectors, 'key_to_index', set())

    # Step 1: Tokenize corpus
    corpus_tokens = [token.lower() for doc in corpus_docs for token in word_tokenize(doc)]
    if not corpus_tokens:
        return 0.0

    # Step 2: Co-occurrence counts
    cooccurrence = defaultdict(int)
    word_counts = Counter(corpus_tokens)
    total_windows = 0

    for i in range(len(corpus_tokens)):
        window = corpus_tokens[i+1:i+1+window_size]
        for w1 in [corpus_tokens[i]]:
            for w2 in window:
                if w1 != w2:
                    pair = tuple(sorted((w1, w2)))
                    cooccurrence[pair] += 1
        total_windows += 1
    total_word_count = sum(word_counts.values())

    # Step 3: Get top words per cluster
    top_words_per_cluster = get_top_n_nearest_points(gmm, embedding_model, corpus_tokens, top_n)

    # Step 4: Compute NPMI per cluster
    def npmi_from_counts(word_list):
        if len(word_list) < 2:
            return 0
        scores = []
        for w1, w2 in combinations(word_list, 2):
            if not has_word(w1) or not has_word(w2):
                continue
            pair = tuple(sorted((w1, w2)))
            count_joint = cooccurrence.get(pair, 0) + 1e-10
            p_joint = count_joint / (total_windows + 1e-10)
            p1 = (word_counts.get(w1,0) + 1e-10) / (total_word_count + 1e-10)
            p2 = (word_counts.get(w2,0) + 1e-10) / (total_word_count + 1e-10)
            if p_joint > 0:
                pmi = np.log(p_joint / (p1 * p2))
                npmi = pmi / (-np.log(p_joint)) if p_joint < 1 else 0
                scores.append(npmi)
        return np.mean(scores) if scores else 0

    npmi_scores = [npmi_from_counts(words) for words in top_words_per_cluster]
    return np.mean(npmi_scores) if npmi_scores else 0.0
