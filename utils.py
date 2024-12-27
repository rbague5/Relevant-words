import ast
import json
import logging
import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations
from matplotlib import pyplot as plt
from gensim.models import Word2Vec
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
)
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cdist

from callback import MonitorLoss
from config import rating_threshold, top_n_nearest_points

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


def train_w2v_model(model_path, model_name, corpus):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    path = os.path.join(model_path, model_name)
    if not os.path.exists(path):
        monitor_loss = MonitorLoss()
        model = Word2Vec(
            sentences=corpus,
            vector_size=100,
            window=5,
            min_count=2,
            workers=4,
            sg=1,
            epochs=20,
            compute_loss=True,  # Enable loss computation
            callbacks=[monitor_loss],
        )
        model.save(path)
    else:
        model = Word2Vec.load(path)
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
    w, h = n_points, len(gmm_model.means_)
    top_n_list = [[0 for x in range(w)] for y in range(h)]
    for n in range(w):
        embedding_corpus = np.array([w2v_model.wv[key] for key in corpus])
        if len(embedding_corpus) == 0:
            break
        closest_idx, _ = pairwise_distances_argmin_min(
            gmm_model.means_, embedding_corpus
        )
        closest_words = [corpus[idx] for idx in closest_idx.tolist()]
        for idx, val in enumerate(closest_words):
            top_n_list[idx][n] = val
        corpus = list(filter(lambda x: x not in closest_words, corpus))
    return top_n_list


def calculate_semantic_coherence(w2v_model, gmm, corpus_words, top_n=5):
    coherence_scores = []
    for idx in range(len(gmm.means_)):
        cluster_words = get_top_n_nearest_points(gmm, w2v_model, corpus_words, top_n)[
            idx
        ]
        similarities = []
        for word1, word2 in combinations(cluster_words, 2):
            if word1 in w2v_model.wv and word2 in w2v_model.wv:
                sim = w2v_model.wv.similarity(word1, word2)
                similarities.append(sim)
        coherence_scores.append(np.mean(similarities) if similarities else 0)
    return np.mean(coherence_scores)


def calculate_perplexity(gmm):
    # Perplexity is a commonly used metric for language models; for GMMs, it can be adapted
    # as the exponential of the average negative log-likelihood.
    log_likelihood = gmm.score(
        gmm.means_
    )  # Mean of each component as representative point
    return np.exp(-log_likelihood)


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
    npmi_scores = []
    total_word_count = sum(
        w2v_model.wv.get_vecattr(word, "count")
        for word in corpus_words
        if word in w2v_model.wv
    )

    for idx in range(len(gmm.means_)):
        cluster_words = get_top_n_nearest_points(gmm, w2v_model, corpus_words, top_n)[
            idx
        ]
        pair_scores = []
        for word1, word2 in combinations(cluster_words, 2):
            if word1 in w2v_model.wv and word2 in w2v_model.wv:
                # Calculate probabilities based on word frequency
                p_word1 = w2v_model.wv.get_vecattr(word1, "count") / total_word_count
                p_word2 = w2v_model.wv.get_vecattr(word2, "count") / total_word_count
                p_word1_word2 = w2v_model.wv.similarity(word1, word2)

                if p_word1 > 0 and p_word2 > 0 and p_word1_word2 > 0:
                    npmi = np.log(p_word1_word2 / (p_word1 * p_word2)) / -np.log(
                        p_word1_word2
                    )
                    pair_scores.append(npmi)

        npmi_scores.append(np.mean(pair_scores) if pair_scores else 0)

    return np.mean(npmi_scores)


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

    # Find the best cluster for each metric
    for metric in [
        "aic",
        "bic",
        "davies_bouldin_score",
        "perplexity",
        "intra_distance",
    ]:
        # For these metrics, we want the minimum value
        best_clusters[metric] = min(
            cluster_metrics, key=lambda x: cluster_metrics[x][metric]
        )

    for metric in [
        "silhouette_score",
        "calinski_harabasz_score",
        "semantic_coherence",
        "topic_diversity",
        "inter_distance",
        "npmi",
    ]:
        # For these metrics, we want the maximum value
        best_clusters[metric] = max(
            cluster_metrics, key=lambda x: cluster_metrics[x][metric]
        )

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


def train_gmm_model(w2v_model, nouns, model_path, n_clusters_range=range(2, 10)):
    clustering_results = {}
    cluster_metrics = {}
    closest = {}
    corpus = set(w2v_model.wv.index_to_key[:]).intersection(nouns)
    embedding_corpus = np.array(
        [w2v_model.wv[key] for key in corpus]
    )  # Clustering con los sustantivos

    for n_clusters in n_clusters_range:
        model_name = str(n_clusters)
        if not model_saved(model_path, model_name):
            peaks = retrieve_peaks(n_clusters, w2v_model, corpus)
            gmm = GaussianMixture(n_components=len(peaks), means_init=peaks).fit(
                embedding_corpus
            )
            clustering_results[model_name] = gmm
            save_gmm_model(os.path.join(model_path, model_name), gmm)
        else:
            gmm = load_gmm_model(os.path.join(model_path, model_name))
            clustering_results[model_name] = gmm

        top_n_closest_words = get_top_n_nearest_points(
            gmm, w2v_model, list(corpus), top_n_nearest_points
        )
        labels = gmm.predict(embedding_corpus)  # Get cluster labels

        closest[model_name] = top_n_closest_words

        cluster_metrics[model_name] = {
            "aic": gmm.aic(embedding_corpus),
            "bic": gmm.bic(embedding_corpus),
            "silhouette_score": (
                silhouette_score(embedding_corpus, labels)
                if len(set(labels)) > 1
                else None
            ),
            "calinski_harabasz_score": (
                calinski_harabasz_score(embedding_corpus, labels)
                if len(set(labels)) > 1
                else None
            ),
            "davies_bouldin_score": (
                davies_bouldin_score(embedding_corpus, labels)
                if len(set(labels)) > 1
                else None
            ),
            "semantic_coherence": calculate_semantic_coherence(
                w2v_model, gmm, list(corpus), top_n_nearest_points
            ),
            "perplexity": calculate_perplexity(gmm),
            "topic_diversity": calculate_topic_diversity(
                w2v_model, gmm, list(corpus), top_n_nearest_points
            ),
            "intra_distance": calculate_intra_distances(gmm, embedding_corpus),
            "inter_distance": calculate_inter_distances(gmm, embedding_corpus),
            "npmi": calculate_npmi(gmm, list(corpus), w2v_model),
        }

        logger.info(f"Cluster {model_name} metrics: {cluster_metrics[model_name]}")

    return clustering_results, cluster_metrics, closest


def retrieve_peaks(n_peaks, w2v_model, corpus):
    peaks = []
    last_index_found = 0
    for i in range(n_peaks):
        while last_index_found < len(w2v_model.wv.index_to_key):
            candidate_peak = w2v_model.wv.index_to_key[last_index_found]
            if candidate_peak in corpus:
                peaks.append(w2v_model.wv[candidate_peak])
                last_index_found += 1
                break
            last_index_found += 1
    return peaks


def retrieve_best_gmm_model(aic_bic_results):
    results_df = pd.DataFrame.from_dict(aic_bic_results, orient="index")
    results_df.columns = ["aic", "bic"]
    return results_df[results_df["aic"] == results_df["aic"].min()].index[0]


def retrieve_best_model_results(
    best_gmm_model_name, trained_models, w2v_model, closest_words
):
    n_clusters = best_gmm_model_name
    model = trained_models[best_gmm_model_name]
    embedding_corpus = np.array(
        [w2v_model.wv[key] for key in np.array(closest_words).flatten().tolist()]
    )
    labels = np.indices(np.array(closest_words).shape)[0].flatten().tolist()
    # labels = model.predict(embedding_corpus)
    probabilities = model.score_samples(embedding_corpus)
    # probabilities = normalize(probabilities[:, np.newaxis], axis=0).ravel() #TODO revisar normalización de logProbabilities
    sample = np.array(closest_words).flatten()
    return probabilities, get_words_by_cluster(sample, labels, n_clusters), labels


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


# def perform_tsne(w2v_model, nouns, labels, figure_path):
#     plt.figure(figsize=(15, 10))
#     palette = sns.color_palette("bright", 10)
#     tsne = TSNE(n_components=2, random_state=0)
#     embedding_corpus = np.array([w2v_model.wv[key] for key in set(w2v_model.wv.index_to_key[:]).intersection(nouns)])
#     X_embedded = tsne.fit_transform(X=embedding_corpus)
#     ax = sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, legend='full',
#                          palette=palette[:len(set(labels))])
#     if not os.path.exists(figure_path):
#         os.makedirs(figure_path)
#     a = pd.concat({'x': pd.Series(X_embedded[:, 0]), 'y': pd.Series(X_embedded[:, 1]), 'val': pd.Series(np.array(set(w2v_model.wv.index_to_key[:]).intersection(nouns)))}, axis=1)
#     for i, point in a.iterrows():
#         plt.gca().text(point['x']+.02, point['y'], str(point['val']))
#     plt.savefig(os.path.join(figure_path, "topics.png"))
def perform_tsne(w2v_model, labels, closest_words, figure_path, review_type):
    plt.figure(figsize=(15, 10))
    palette = sns.color_palette("bright", 30)

    embedding_corpus = np.array(
        [w2v_model.wv[key] for key in np.array(closest_words).flatten().tolist()]
    )
    perplexity = min(30, len(embedding_corpus) - 1)
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    X_embedded = tsne.fit_transform(X=embedding_corpus)
    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=labels,
        legend="full",
        style=labels,
        palette=palette[: len(set(labels))],
    )

    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    for label, x, y in zip(
        np.array(closest_words).flatten().tolist(), X_embedded[:, 0], X_embedded[:, 1]
    ):
        plt.annotate(
            label, xy=(x, y), xytext=(0, 0), fontsize=10, textcoords="offset points"
        )
    plt.savefig(os.path.join(figure_path, review_type + ".png"))
    plt.close()


def save_topic_clusters_results(cluster_dict, results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for key in cluster_dict:
        np.save(os.path.join(results_path, str(key) + ".npy"), cluster_dict[key])


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
        topic_clusters[filename.split(".")[0]] = np.load(
            os.path.join(results_path, filename)
        )
    return topic_clusters
