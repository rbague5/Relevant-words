import ast
import json
import logging
import math
import os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, pairwise_distances
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from config import rating_threshold, top_n_nearest_points
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from nltk.corpus import wordnet as wn

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

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    _model_cache[model_name] = model
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


def get_top_n_nearest_points(gmm_model, w2v_model, corpus, n_points):
    # Usar la interfaz correcta
    keyed_vectors = w2v_model.wv if hasattr(w2v_model, 'wv') else w2v_model

    # Filtrar palabras fuera del vocabulario
    valid_corpus = [word for word in corpus if word in keyed_vectors]
    if not valid_corpus:
        return []

    word_vectors = np.array([keyed_vectors[word] for word in valid_corpus])
    distances = pairwise_distances(gmm_model.means_, word_vectors)  # shape: (n_clusters, n_words)

    used_words = set()
    top_n_list = []

    for cluster_distances in distances:
        cluster_words = []
        for idx in np.argsort(cluster_distances):
            if valid_corpus[idx] not in used_words:
                cluster_words.append(valid_corpus[idx])
                used_words.add(valid_corpus[idx])
            if len(cluster_words) >= n_points:
                break
        top_n_list.append(cluster_words)

    return top_n_list



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
            best_clusters[metric] = best_cluster
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
            best_clusters[metric] = best_cluster
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


def train_gmm_model(w2v_model, nouns, model_path, n_clusters_range=range(3, 10), top_n_points=10):
    from metrics import calculate_semantic_coherence, calculate_intra_distances, calculate_inter_distances, calculate_npmi, calculate_scci

    clustering_results = {}
    cluster_metrics = {}
    closest = {}

    # Obtener KeyedVectors independientemente del tipo de modelo
    keyed_vectors = w2v_model.wv if hasattr(w2v_model, 'wv') else w2v_model

    # Filtrar sustantivos presentes en el vocabulario
    corpus = sorted(set(keyed_vectors.index_to_key).intersection(nouns))
    embedding_corpus = np.array([keyed_vectors[word] for word in corpus])

    for n_clusters in n_clusters_range:
        model_name = str(n_clusters)
        model_file_path = os.path.join(model_path, model_name)

        if not model_saved(model_path, model_name):
            peaks = retrieve_peaks(n_clusters, keyed_vectors, corpus)
            gmm = GaussianMixture(n_components=n_clusters, means_init=peaks, random_state=42)
            gmm.fit(embedding_corpus)
            save_gmm_model(model_file_path, gmm)
        else:
            gmm = load_gmm_model(model_file_path)

        clustering_results[model_name] = gmm

        labels = gmm.predict(embedding_corpus)
        top_n_closest_words = get_top_n_nearest_points(gmm, keyed_vectors, corpus, top_n_points)

        closest[model_name] = top_n_closest_words

        metrics = {
            "silhouette_score": silhouette_score(embedding_corpus, labels) if len(set(labels)) > 1 else None,
            "semantic_coherence": calculate_semantic_coherence(keyed_vectors, gmm, corpus, top_n_points),
            "intra_distance": calculate_intra_distances(gmm, embedding_corpus),
            "inter_distance": calculate_inter_distances(gmm, embedding_corpus),
            "npmi": calculate_npmi(gmm, corpus, keyed_vectors),
            "scci": calculate_scci(embedding_corpus, labels)
        }

        cluster_metrics[model_name] = metrics
        logger.info(f"Cluster {model_name} metrics: {metrics}")

    return clustering_results, cluster_metrics, closest


def retrieve_peaks(n_peaks, embedding_model, corpus):
    peaks = []

    # Usar el acceso correcto dependiendo del tipo de modelo
    if hasattr(embedding_model, 'wv'):
        keyed_vectors = embedding_model.wv
    else:
        keyed_vectors = embedding_model

    last_index_found = 0
    index_keys = keyed_vectors.index_to_key

    for i in range(n_peaks):
        while last_index_found < len(index_keys):
            candidate_peak = index_keys[last_index_found]
            last_index_found += 1  # Mover fuera del if para evitar repetir
            if candidate_peak in corpus:
                peaks.append(keyed_vectors[candidate_peak])
                break

    return peaks


def retrieve_best_gmm_model(aic_bic_results):
    results_df = pd.DataFrame.from_dict(aic_bic_results, orient="index")
    results_df.columns = ["aic", "bic"]
    return results_df[results_df["aic"] == results_df["aic"].min()].index[0]


def retrieve_best_model_results(best_gmm_model_name, trained_models, w2v_model, closest_words):
    model = trained_models[best_gmm_model_name]
    n_clusters = len(model.means_)

    # Obtener vectorizador adecuado
    if hasattr(w2v_model, 'wv'):
        keyed_vectors = w2v_model.wv
    else:
        keyed_vectors = w2v_model

    # Flatten + filtrar palabras fuera del vocabulario
    flat_words = [word for word in np.array(closest_words).flatten().tolist() if word in keyed_vectors]
    if not flat_words:
        raise ValueError("No valid words found in the model vocabulary.")

    # Embeddings y etiquetas
    embedding_corpus = np.array([keyed_vectors[word] for word in flat_words])
    labels = np.indices(np.array(closest_words).shape)[0].flatten().tolist()[:len(embedding_corpus)]

    # Predicción de probabilidad logarítmica por muestra
    probabilities = model.score_samples(embedding_corpus)

    # Alternativa: cluster real de cada punto
    # labels = model.predict(embedding_corpus)

    # Resultado agrupado por clusters
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


def perform_tsne(w2v_model, labels, closest_words, figure_path, review_type, metric, n_iter=3000):
    # Soporte para modelos con o sin `.wv`
    keyed_vectors = w2v_model.wv if hasattr(w2v_model, "wv") else w2v_model

    # Filtrar palabras válidas (en el vocabulario)
    flat_words = np.array(closest_words).flatten().tolist()
    filtered = [(word, keyed_vectors[word]) for word in flat_words if word in keyed_vectors]

    if not filtered:
        raise ValueError("No valid words found in the model vocabulary.")

    words, embeddings = zip(*filtered)
    embedding_corpus = np.array(embeddings)

    # Asegurar que perplexity sea válida
    n_samples = len(embedding_corpus)
    perplexity = min(max(5, n_samples // 3), 30)

    # t-SNE
    tsne = TSNE(n_components=2, n_iter=n_iter, random_state=42, perplexity=perplexity)
    X_embedded = tsne.fit_transform(X=embedding_corpus)

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

    # Crear carpeta si no existe
    os.makedirs(figure_path, exist_ok=True)

    # Anotar cada punto
    for label, x, y in zip(words, X_embedded[:, 0], X_embedded[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), fontsize=9, textcoords="offset points")

    fig_path = figure_path + metric + "_" + review_type + "_" + f"{str(len(set(labels)))}.png"
    # Guardar figura
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    # Filter perplexities valid for this dataset
    # valid_perplexities = [p for p in [5,15,30,50] if p < n_samples]
    #
    # if not valid_perplexities:
    #     raise ValueError(f"No valid perplexities for n_samples={n_samples}")
    #
    # # Optional: generate a t-SNE plot for each perplexity
    # for perplexity in valid_perplexities:
    #     tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
    #     X_embedded = tsne.fit_transform(X=embedding_corpus)
    #
    #     plt.figure(figsize=(15, 10))
    #     palette = sns.color_palette("bright", max(len(set(labels)), 10))
    #
    #     sns.scatterplot(
    #         x=X_embedded[:, 0],
    #         y=X_embedded[:, 1],
    #         hue=labels[:n_samples],
    #         palette=palette[:len(set(labels))],
    #         legend="full",
    #         style=labels[:n_samples],
    #     )
    #
    #     # # Annotate points
    #     # for label, x, y in zip(words, X_embedded[:, 0], X_embedded[:, 1]):
    #     #     plt.annotate(label, xy=(x, y), xytext=(2, 2), fontsize=9, textcoords="offset points")
    #
    #     os.makedirs(figure_path, exist_ok=True)
    #     plt.title(f"t-SNE Visualization (perplexity={perplexity}, n_iter={n_iter}) - {review_type} ({metric})")
    #     plt.savefig(os.path.join(figure_path, f"tsne_{review_type}_{metric}_perp{perplexity}.png"))
    #     plt.close()

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
