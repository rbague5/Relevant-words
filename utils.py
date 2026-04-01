import ast
import json
import logging
import math
import os
import pickle
from collections import defaultdict, Counter
from heapq import heappush, heappushpop
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from config import rating_threshold
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from nltk.corpus import wordnet as wn

from new_scci import calculate_scci_gmm

_model_cache = {}  # Global model cache

sns.set(rc={"figure.figsize": (11.7, 8.27)})
palette = sns.color_palette("bright", 10)


def instantiate_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = instantiate_logger()

class ContextualEmbeddingWrapper:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        # GPU Support
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_word_embedding(self, word: str):
        # Encode the word as a "sentence"
        with torch.no_grad():
            emb = self.model.encode([word], convert_to_numpy=True)[0]
        return emb

    def __repr__(self):
        return f"ContextualEmbeddingWrapper({self.model})"

def retrieve_word_frequencies(data):
    cv = CountVectorizer(min_df=0.10)
    dict_results = {"itemId": [], "word": [], "frequency": []}
    vectorized_results = cv.fit_transform(data)
    for restaurant_idx, results in enumerate(vectorized_results):
        freq = results.data
        idx_vocabulary = results.indices
        for i, j in zip(freq, idx_vocabulary):
            dict_results["itemId"].append(data.index[restaurant_idx])
            dict_results["word"].append(
                list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(j)]
            )
            dict_results["frequency"].append(i)
    res = pd.DataFrame().from_dict(dict_results)
    res.sort_values(["itemId", "frequency"], ascending=[True, False], inplace=True)
    return res


def flatten_reviews_by_restaurant(data):
    data_grouped_by_restaurant = data.groupby("itemId").agg(
        {
            "nouns": lambda x: [
                " ".join(ast.literal_eval(w)) for w in list((np.hstack(x)))
            ]
        }
    )
    data_grouped_by_restaurant["nouns"] = data_grouped_by_restaurant["nouns"].apply(
        lambda x: " ".join(x)
    )
    return data_grouped_by_restaurant


def retrieve_word_frequencies_by_review(data):
    tf_positive = retrieve_word_frequencies(
        flatten_reviews_by_restaurant(data[data["rating"] >= rating_threshold])["nouns"]
    )
    tf_negative = retrieve_word_frequencies(
        flatten_reviews_by_restaurant(data[data["rating"] < rating_threshold])["nouns"]
    )
    return tf_positive, tf_negative


def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100


def load_w2v_model(model_path, model_name):
    return Word2Vec.load(os.path.join(model_path, model_name))


def train_or_load_embedding_model(model_name, model_path=None, corpus=None):
    global _model_cache

    logger.info(f"Using model: {model_name}")

    # Ensure a valid path is provided, otherwise default to the downloaded models path
    downloaded_models_path = model_path or "./models/"
    if not os.path.exists(downloaded_models_path):
        os.makedirs(downloaded_models_path)

    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name.lower() == "word2vec":
        path = os.path.join(downloaded_models_path, model_name)

        if not os.path.exists(path):
            if corpus is None:
                raise ValueError("Corpus is required to train Word2Vec.")
            monitor_loss = MonitorLoss()
            model = Word2Vec(
                sentences=corpus,
                vector_size=100,
                window=5,
                min_count=2,
                workers=4,
                sg=1,
                epochs=20,
                compute_loss=True,
                callbacks=[monitor_loss],
            )
            model.save(path)
        else:
            model = Word2Vec.load(path)
    elif model_name.lower() == "fasttext":
        path = os.path.join(downloaded_models_path, "fasttext-wiki-news-subwords-300")

        # Check if pre-trained model is not in the local directory and download it
        if not os.path.exists(path):
            logger.info("Downloading FastText model...")
            model = api.load("fasttext-wiki-news-subwords-300")  # returns KeyedVectors
            model.save(path)
        else:
            model = KeyedVectors.load(path)
    elif model_name.lower() == "google":
        path = os.path.join(downloaded_models_path, "word2vec-google-news-300")

        # Check if pre-trained model is not in the local directory and download it
        if not os.path.exists(path):
            logger.info("Downloading Google News Word2Vec model...")
            model = api.load("word2vec-google-news-300")  # returns KeyedVectors
            model.save(path)
        else:
            model = KeyedVectors.load(path)
    elif model_name.lower() == "glove":
        model = api.load("glove-wiki-gigaword-300")
    elif model_name.lower() == "minilm":
        model = ContextualEmbeddingWrapper("sentence-transformers/all-MiniLM-L6-v2")
    elif model_name.lower() == "sbert":
        model = ContextualEmbeddingWrapper("sentence-transformers/all-roberta-large-v1")
    elif model_name.lower() == "e5":
        model = ContextualEmbeddingWrapper("intfloat/multilingual-e5-large")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    _model_cache[model_name] = model
    logger.info(f"Loaded model: {model_name}")
    return model



def save_gmm_model(model_path, model):
    logger.info("Saving... " + model_path)
    np.save(model_path + "_weights", model.weights_, allow_pickle=False)
    np.save(model_path + "_means", model.means_, allow_pickle=False)
    np.save(model_path + "_covariances", model.covariances_, allow_pickle=False)


def load_gmm_model(model_path):
    logger.info("Loading... " + model_path)
    means = np.load(model_path + "_means.npy")
    covar = np.load(model_path + "_covariances.npy")
    loaded_gmm = GaussianMixture(n_components=len(means), covariance_type="full")
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(model_path + "_weights.npy")
    loaded_gmm.means_ = means
    loaded_gmm.covariances_ = covar
    return loaded_gmm


def model_saved(model_path, model_name):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for filename in os.listdir(model_path):
        if filename.startswith(model_name):
            return True
    return False


def plot_elbow(aic_bic_results):
    n_clusters = sorted(int(k) for k in aic_bic_results.keys())
    aic_values = [aic_bic_results[str(k)][0] for k in n_clusters]
    bic_values = [aic_bic_results[str(k)][1] for k in n_clusters]

    plt.figure(figsize=(12, 6))
    plt.plot(n_clusters, aic_values, marker="o", label="AIC")
    plt.plot(n_clusters, bic_values, marker="x", label="BIC")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Score")
    plt.title("AIC and BIC Scores for Different GMM Cluster Sizes")
    plt.legend()
    # plt.savefig(os.path.join(topics_path_images, review_type+".png"))


def find_elbow_point(scores):
    # Compute the differences to locate the elbow
    differences = np.diff(scores)
    elbow_index = (
        np.argmin(differences) + 1
    )  # +1 due to diff reducing the array size by 1
    return elbow_index + 1  # Adjust for index offset


def get_top_n_nearest_points(gmm_model, embedding_corpus, corpus, n_points=10):
    """
    Same logic as original:
    - Assign each word to best cluster
    - Take top N per cluster
    - Rebalance deficits
    BUT fully vectorized (no embedding_model needed)
    """

    if len(corpus) == 0:
        return []

    n_clusters = len(gmm_model.means_)

    # 🔥 Compute ALL similarities at once
    sim_matrix = cosine_similarity(embedding_corpus, gmm_model.means_)
    # shape: (n_words, n_clusters)

    # --- Phase 1: Assign each word to its best cluster ---
    best_clusters = np.argmax(sim_matrix, axis=1)

    cluster_words = [[] for _ in range(n_clusters)]
    assigned_words = set()

    for word_idx, cluster_idx in enumerate(best_clusters):
        sim = sim_matrix[word_idx, cluster_idx]
        word = corpus[word_idx]
        cluster_words[cluster_idx].append((sim, word))

    # Sort each cluster by similarity (descending)
    for i in range(n_clusters):
        cluster_words[i].sort(reverse=True)

    # --- Phase 2: Take top N per cluster ---
    final_clusters = [[] for _ in range(n_clusters)]

    for i in range(n_clusters):
        for sim, word in cluster_words[i][:n_points]:
            final_clusters[i].append(word)
            assigned_words.add(word)

    # --- Phase 3: Balance clusters (same logic) ---
    deficit_clusters = [
        (i, n_points - len(final_clusters[i]))
        for i in range(n_clusters)
        if len(final_clusters[i]) < n_points
    ]

    if deficit_clusters:
        # Build candidate pool (same as before)
        candidate_indices = [
            i for i, word in enumerate(corpus)
            if word not in assigned_words
        ]

        for cluster_idx, deficit in deficit_clusters:
            # Get similarities for this cluster
            sims = sim_matrix[candidate_indices, cluster_idx]

            # Sort candidates by similarity
            sorted_idx = np.argsort(sims)[::-1]

            assigned_count = 0
            for idx in sorted_idx:
                word = corpus[candidate_indices[idx]]

                if word not in assigned_words:
                    final_clusters[cluster_idx].append(word)
                    assigned_words.add(word)
                    assigned_count += 1

                if assigned_count >= deficit:
                    break

    return final_clusters


def select_best_clusters(cluster_metrics):
    """
    Selection Criteria for Each Metric
    AIC (Akaike Information Criterion): Lower values indicate better models (less complexity and better fit).
    BIC (Bayesian Information Criterion): Like AIC, lower values are preferable.
    Silhouette Score: Higher values indicate well-separated and compact clusters.
    Calinski-Harabasz Score: Higher values indicate better-defined clusters.
    Davies-Bouldin Score: Lower values are preferable, indicating more compact clusters with good separation.
    Semantic Coherence: Higher scores suggest better semantic similarity within clusters.
    Perplexity: Lower values are preferable, indicating the model’s effectiveness in capturing patterns.
    Topic Diversity: Higher scores indicate more diversity within clusters.
    Intra Distances: Lower intra-cluster and higher inter-cluster distances indicate better-defined clusters.
    NPMI (Normalized Pointwise Mutual Information): Higher values indicate more meaningful clustering in terms of word associations.
    """
    best_clusters = {}

    # List of metrics with the expected operation (min or max)
    min_metrics = ["aic", "bic", "davies_bouldin_score", "perplexity", "intra_distance"]
    max_metrics = ["silhouette_score", "calinski_harabasz_score", "semantic_coherence", "topic_diversity",
                   "inter_distance", "npmi", "scci"]

    # Find the best cluster for each min-metric
    for metric in min_metrics:
        # Prepare a list of clusters that have the current metric and its value is not None
        valid_clusters = [
            cluster for cluster in cluster_metrics
            if metric in cluster_metrics[cluster] and cluster_metrics[cluster][metric] is not None
        ]

        if valid_clusters:
            best_cluster = min(valid_clusters, key=lambda x: cluster_metrics[x][metric])
            best_value = cluster_metrics[best_cluster][metric]
            best_clusters[metric] = {best_cluster: best_value}
        else:
            best_clusters[metric] = None

    # Find the best cluster for each max-metric
    for metric in max_metrics:
        valid_clusters = [
            cluster for cluster in cluster_metrics
            if metric in cluster_metrics[cluster] and cluster_metrics[cluster][metric] is not None
        ]

        if valid_clusters:
            best_cluster = max(valid_clusters, key=lambda x: cluster_metrics[x][metric])
            best_value = cluster_metrics[best_cluster][metric]
            best_clusters[metric] = {best_cluster: best_value}
        else:
            best_clusters[metric] = None

    return best_clusters


def filter_clusters(cluster_metrics, thresholds):
    """
    Filter clusters based on thresholds for specific metrics.
    :param cluster_metrics: Dictionary of cluster metrics.
    :param thresholds: Dictionary specifying minimum (or maximum) thresholds for metrics.
    """
    filtered_clusters = {}
    for cluster, metrics in cluster_metrics.items():
        if all(
            metrics.get(metric, 0) >= thresholds.get(metric, float("-inf"))
            for metric in thresholds
        ):
            filtered_clusters[cluster] = metrics
    return filtered_clusters


def select_overall_best_cluster(cluster_metrics, weights=None):
    """
    Select the best cluster based on weighted combination of metrics.
    :param cluster_metrics: Dictionary of cluster metrics.
    :param weights: Dictionary of metric weights, defaulting to equal weighting.
    """
    if weights is None:
        weights = {
            metric: 1 for metric in cluster_metrics[next(iter(cluster_metrics))].keys()
        }

    overall_scores = {}
    for cluster, metrics in cluster_metrics.items():
        overall_scores[cluster] = sum(
            metrics[metric] * weights.get(metric, 1) for metric in metrics
        )

    # Select the cluster with the highest overall score
    return max(overall_scores, key=overall_scores.get)

def extract_embeddings(embedding_model, nouns, save_path=None):
    """
    Extract embeddings for a list of nouns from any supported embedding model.
    If save_path is provided, it will try to load precomputed embeddings from disk.
    If not found, it will compute and save them.

    Args:
        embedding_model: any supported embedding model (Word2Vec, FastText, GloVe, SBERT, E5, MiniLM)
        nouns: list of words to extract embeddings for
        save_path: optional path to save/load embeddings (.npz)

    Returns:
        corpus: list of words found in the model
        embedding_corpus: np.ndarray of shape (len(corpus), embedding_dim)
    """
    # --- Try to load from disk if save_path exists ---
    if save_path and os.path.exists(save_path):
        data = np.load(save_path, allow_pickle=True)
        corpus = data['corpus'].tolist()
        embedding_corpus = data['embeddings']
        logger.info(f"Loaded embeddings from {save_path}")
        return corpus, embedding_corpus

    # --- Prepare corpus and vector retrieval function ---
    if hasattr(embedding_model, 'wv'):
        # Word2Vec / FastText / GloVe
        keyed_vectors = embedding_model.wv
        vocab = set(keyed_vectors.index_to_key)
        corpus = sorted(vocab.intersection(nouns))
        embedding_corpus = np.vstack([keyed_vectors[word] for word in corpus]).astype(np.float32)

    elif hasattr(embedding_model, 'get_word_embedding'):
        # Transformer-based models: SBERT / E5 / MiniLM
        corpus = sorted(set(nouns))
        embedding_list = [embedding_model.get_word_embedding(word) for word in corpus]
        embedding_corpus = np.vstack(embedding_list).astype(np.float32)

    else:
        # Fallback: keyed vector wrapper (e.g., GloVe)
        vocab = set(embedding_model.index_to_key)
        corpus = sorted(vocab.intersection(nouns))
        embedding_corpus = np.vstack([embedding_model[word] for word in corpus]).astype(np.float32)

    # --- Save to disk if save_path is provided ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, corpus=corpus, embeddings=embedding_corpus)
        logger.info(f"Saved embeddings to {save_path}")

    return corpus, embedding_corpus

def train_gmm_model(embedding_corpus, corpus, model_path, n_clusters_range=range(3, 10), top_n_points=10, covariance_type='diag',
                reg_covar=1e-5, max_iter=200, n_init=3):
    from metrics import calculate_semantic_coherence, calculate_intra_distances, calculate_inter_distances
    from new_scci import calculate_scci_improved

    clustering_results = {}
    cluster_metrics = {}
    closest = {}


    for n_clusters in n_clusters_range:
        model_name = str(n_clusters)
        model_file_path = os.path.join(model_path, model_name)

        if not model_saved(model_path, model_name):
            peaks = retrieve_peaks(n_clusters, corpus, embedding_corpus)
            gmm = GaussianMixture(
                n_components=n_clusters,
                means_init=peaks,
                random_state=42,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                max_iter=max_iter,
                n_init=n_init,
            )
            gmm.fit(embedding_corpus)
            save_gmm_model(model_file_path, gmm)
        else:
            gmm = load_gmm_model(model_file_path)

        clustering_results[model_name] = gmm

        labels = gmm.predict(embedding_corpus)
        top_n_closest_words = get_top_n_nearest_points(gmm, embedding_corpus, corpus, top_n_points)

        closest[model_name] = top_n_closest_words

        sil_score = silhouette_score(embedding_corpus, labels) if len(set(labels)) > 1 else None
        semantic_coherence = calculate_semantic_coherence(embedding_corpus, corpus, top_words_per_cluster=top_n_closest_words)
        intra_distance = calculate_intra_distances(gmm, embedding_corpus)
        inter_distance = calculate_inter_distances(gmm, embedding_corpus)
        # npmi = calculate_npmi(gmm, corpus, keyed_vectors, top_n=top_n_points)
        scci = calculate_scci_improved(embedding_corpus, labels, normalize_output=True)
        # gmm_scci = calculate_scci_gmm(embedding_corpus, gmm)

        metrics = {
            "silhouette_score": round(sil_score, 4),
            "semantic_coherence": round(semantic_coherence, 4),
            "intra_distance": round(intra_distance, 4),
            "inter_distance": round(inter_distance, 4),
            # "npmi": round(npmi, 4),
            "scci": scci,
            # "gmm_scci": gmm_scci
        }

        cluster_metrics[model_name] = metrics
        logger.info(f"Cluster {model_name} metrics: {metrics}")

    return clustering_results, cluster_metrics, closest


def retrieve_peaks(n_clusters, corpus, embedding_corpus):
    """
    Selects initial GMM cluster centers ('peaks') based on the n_clusters
    most frequent words in the corpus. Returns their embeddings.

    Parameters
    ----------
    n_clusters : int
        Number of GMM clusters.
    corpus : list[str]
        List of all words in the corpus (aligned with embedding_corpus).
    embedding_corpus : np.ndarray
        Precomputed embeddings, shape (len(corpus), dim)

    Returns
    -------
    peaks : np.ndarray
        Initial cluster centers for GMM, shape (n_clusters, embedding_dim)
    """

    # Count word frequencies
    word_counts = Counter(corpus)

    # Select the top-n_clusters most frequent words
    top_words = [word for word, _ in word_counts.most_common(n_clusters)]

    # Map words to indices in corpus
    word_to_idx = {word: i for i, word in enumerate(corpus)}

    # Collect embeddings for the top words
    peaks = []
    for word in top_words:
        idx = word_to_idx.get(word)
        if idx is not None:
            peaks.append(embedding_corpus[idx])

    # If there are fewer peaks than n_clusters, pad randomly
    if len(peaks) < n_clusters:
        remaining_idx = [i for i in range(len(corpus)) if corpus[i] not in top_words]
        for i in range(n_clusters - len(peaks)):
            peaks.append(embedding_corpus[remaining_idx[i % len(remaining_idx)]])

    return np.array(peaks)


def retrieve_best_gmm_model(aic_bic_results):
    results_df = pd.DataFrame.from_dict(aic_bic_results, orient="index")
    results_df.columns = ["aic", "bic"]
    return results_df[results_df["aic"] == results_df["aic"].min()].index[0]


def retrieve_best_model_results(best_gmm_model_name, trained_models, embedding_corpus, corpus, closest_words):
    """
    Retrieve log-likelihoods, clustered words, and labels for the best GMM model.
    Uses precomputed embeddings (`embedding_corpus`) and word list (`corpus`).

    Parameters:
        best_gmm_model_name: str, key of the trained GMM model
        trained_models: dict of {model_name: GaussianMixture instance}
        embedding_corpus: np.ndarray of shape (num_words, embedding_dim)
        corpus: list of words corresponding to embedding_corpus rows
        closest_words: list of lists of top words per cluster

    Returns:
        probabilities: array of log-likelihoods for each word in embedding_corpus
        clustered_words: dict {cluster_idx: [words]}
        labels: list of cluster indices for each word
    """
    model = trained_models[best_gmm_model_name]
    n_clusters = len(model.means_)

    # --- Map words in closest_words to indices in corpus ---
    word_to_idx = {word: i for i, word in enumerate(corpus)}

    flat_words = []
    labels = []
    for cluster_idx, group in enumerate(closest_words):
        for word in (group if isinstance(group, (list, tuple)) else [group]):
            if word in word_to_idx:
                flat_words.append(word)
                labels.append(cluster_idx)

    if not flat_words:
        raise ValueError("No valid words from closest_words found in corpus.")

    # --- Extract embeddings for these words ---
    indices = [word_to_idx[w] for w in flat_words]
    embedding_subset = embedding_corpus[indices]

    # --- Compute log-likelihoods under GMM ---
    probabilities = model.score_samples(embedding_subset)

    # --- Group words by cluster ---
    clustered_words = get_words_by_cluster(flat_words, labels, n_clusters)

    return probabilities, clustered_words, labels

def save_cluster_metrics(metrics, file_path):
    def _convert_to_serializable(obj):
        """Recursively convert numpy objects to native Python types."""
        if isinstance(obj, dict):
            return {k: _convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_convert_to_serializable(x) for x in obj]
        elif isinstance(obj, np.generic):  # Handles all numpy scalar types
            return obj.item()
        else:
            return obj

    try:
        file_path = file_path + "_metrics.json"
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        serializable_metrics = _convert_to_serializable(metrics)
        # Write the dictionary to a JSON file
        with open(file_path, "w") as f:
            json.dump(serializable_metrics, f, indent=4)

        logger.info(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving metrics: {e}")


def get_words_by_cluster(sample, labels, n_clusters):
    clusters = {}
    labels = np.array(labels)
    sample = np.array(sample)

    for id_cluster in range(int(n_clusters)):
        cluster_words = sample[np.where(labels == id_cluster)[0]]
        clusters[id_cluster] = cluster_words
    return clusters


def perform_tsne(embedding_model, labels, closest_words, figure_path, review_type, metric, n_iter=3000, img_name: str = ""):
    # Soporte para modelos con o sin `.wv`
    if hasattr(embedding_model, 'wv'):
        keyed_vectors = embedding_model.wv
    elif hasattr(embedding_model, 'get_word_embedding'):
        class ContextualWrapper:
            def __init__(self, model):
                self.model = model

            def __getitem__(self, word):
                return self.model.get_word_embedding(str(word))

            def __contains__(self, word):
                try:
                    _ = self.model.get_word_embedding(str(word))
                    return True
                except:
                    return False
        keyed_vectors = ContextualWrapper(embedding_model)
    else:
        keyed_vectors = embedding_model

    try:
        flat_words = np.array(closest_words).flatten().tolist()
    except Exception as e:
        flat_words = [w for sublist in closest_words for w in (sublist if isinstance(sublist, list) else [sublist])]

    filtered = [(word, keyed_vectors[word]) for word in flat_words if word in keyed_vectors]
    if not filtered:
        raise ValueError("No valid words found in the model vocabulary.")

    words, embeddings = zip(*filtered)
    embedding_corpus = np.array(embeddings)

    # Asegurar que perplexity sea válida
    n_samples = len(embedding_corpus)
    perplexity = min(max(5, n_samples // 3), 30)
    random_state = 42

    # t-SNE
    tsne = TSNE(n_components=2, n_iter=n_iter, random_state=random_state, perplexity=perplexity)
    X_embedded = tsne.fit_transform(X=embedding_corpus)

    tsne_data = {
        "embedding_corpus": embedding_corpus,
        "X_embedded": X_embedded,
        "words": words,
        "labels": labels,
        "n_samples": n_samples,
        "perplexity": perplexity,
        "n_iter": n_iter,
        "random_state": random_state,
    }

    # Crear carpeta si no existe
    os.makedirs(figure_path, exist_ok=True)

    save_path = figure_path + "plot_pkl_obj/" + f"tsne_inputs_{metric}_{review_type}_{str(len(set(labels)))}.pkl"
    os.makedirs(os.path.join(figure_path, "plot_pkl_obj"), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(tsne_data, f)

    # Configuración de gráfico
    plt.figure(figsize=(15, 10))
    palette = sns.color_palette("bright", max(len(set(labels)), 10))

    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=labels[:n_samples],
        palette=palette[:len(set(labels))],
        legend="full",
        style=labels[:n_samples],
    )

    # Anotar cada punto
    for label, x, y in zip(words, X_embedded[:, 0], X_embedded[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), fontsize=15, textcoords="offset points")

    fig_path = figure_path + metric + "_" + review_type + "_" + f"{str(len(set(labels)))}.png"
    # Guardar figura
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def perform_umap(
    embedding_corpus, labels, words, figure_path, review_type, metric, n_neighbors=15, min_dist=0.1, img_name: str = ""
):
    """
    Perform UMAP on a precomputed embedding_corpus for given words and labels.
    Saves both the UMAP object and a scatterplot.
    """
    n_samples = len(embedding_corpus)
    random_state = 42

    # UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    X_embedded = reducer.fit_transform(embedding_corpus)

    umap_data = {
        "embedding_corpus": embedding_corpus,
        "X_embedded": X_embedded,
        "words": words,
        "labels": labels,
        "n_samples": n_samples,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "random_state": random_state,
    }

    # Crear carpeta si no existe
    os.makedirs(figure_path, exist_ok=True)
    save_path = figure_path + "plot_pkl_obj/" + f"umap_inputs_{metric}_{review_type}_{str(len(set(labels)))}.pkl"
    os.makedirs(os.path.join(figure_path, "plot_pkl_obj"), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(umap_data, f)

    # Configuración de gráfico
    plt.figure(figsize=(15, 10))
    palette = sns.color_palette("bright", max(len(set(labels)), 10))

    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=labels[:n_samples],
        palette=palette[:len(set(labels))],
        legend="full",
        style=labels[:n_samples],
    )

    # Anotar cada punto
    for label, x, y in zip(words, X_embedded[:, 0], X_embedded[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), fontsize=15, textcoords="offset points")

    fig_path = figure_path + metric + "_" + review_type + "_" + f"{str(len(set(labels)))}.png"
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def save_topic_clusters_results(cluster_dict, results_path, class_review, metric):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for key, words in cluster_dict.items():
        file_path = os.path.join(results_path, metric + "_" + class_review)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        with open(os.path.join(file_path, f"{key}.txt"), 'w') as f:
            for word in words:
                f.write(f"{word}\n")


def generate_histogram(df, figure_path, top_n, y_max, type_review=None):
    plt.figure(figsize=(15, 10))
    palette = "Blues_r"
    if type_review is not None:
        if type_review == "positive":
            palette = "BuGn_r"
        elif type_review == "negative":
            palette = "OrRd_r"
    # ax = sns.barplot(x="word", y="frequency", data=df[:top_n], palette=palette)
    ax = sns.barplot(
        x="word",
        y="frequency",
        hue="word",
        data=df[:top_n],
        palette=palette,
        legend=False,
    )

    ax.set_ylabel("count")
    ax.set_ylim([0, y_max])
    path = os.path.join(figure_path, str(df["itemId"].values[0]))
    if not os.path.exists(path):
        os.makedirs(path)
    filename = type_review if type_review is not None else "all_reviews"
    plt.savefig(os.path.join(path, filename + ".png"))

    # https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92


def get_pdf_by_cluster(probabilities, labels):
    df = pd.DataFrame({"prob": probabilities, "label": labels})
    for idx_cluster in np.unique(labels):
        sub_df = df[df["label"] == idx_cluster]
        logger.info(np.sum(np.exp(sub_df["prob"])))


def load_topic_clustes(results_path):
    topic_clusters = {}
    for filename in os.listdir(results_path):
        topic_clusters[filename.split(".")[0]] = np.load(os.path.join(results_path, filename))
    return topic_clusters


def generate_grid_search_params(covariance_types, reg_covars, max_iters, n_inits):
    """
    Generates all combinations of parameter configurations for model training.

    Returns a list of dictionaries, each representing one configuration.
    """
    grid = []
    for cov, reg, max_iter, n_init in product(covariance_types, reg_covars, max_iters, n_inits):
        grid.append({
            'covariance_type': cov,
            'reg_covar': reg,
            'max_iter': max_iter,
            'n_init': n_init
        })
    return grid
