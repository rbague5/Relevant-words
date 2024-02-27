import logging
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from gensim.models import Word2Vec
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.manifold import TSNE
sns.set(rc={'figure.figsize': (11.7, 8.27)})
palette = sns.color_palette("bright", 10)


def instantiate_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = instantiate_logger()


def train_w2v_model(model_path, model_name, corpus):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    path = os.path.join(model_path, model_name)
    if not os.path.exists(path):
        model = Word2Vec(corpus, epochs=15, min_count=10, vector_size=300, window=5, workers=4, sg=1)
        model.save(path)
    else:
        model = Word2Vec.load(path)
    return model


def save_gmm_model(model_path, model):
    logger.info("Saving... " + model_path)
    np.save(model_path + '_weights', model.weights_, allow_pickle=False)
    np.save(model_path + '_means', model.means_, allow_pickle=False)
    np.save(model_path + '_covariances', model.covariances_, allow_pickle=False)


def load_gmm_model(model_path):
    logger.info("Loading... " + model_path)
    means = np.load(model_path + '_means.npy')
    covar = np.load(model_path + '_covariances.npy')
    loaded_gmm = GaussianMixture(n_components=len(means), covariance_type='full')
    loaded_gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    loaded_gmm.weights_ = np.load(model_path + '_weights.npy')
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


def train_gmm_model(w2v_model, reviews_type, nouns, model_path):
    clustering_results = {}
    aic_bic_results = {}
    closest = {}
    corpus = set(w2v_model.wv.index_to_key[:]).intersection(nouns)
    embedding_corpus = np.array([w2v_model.wv[key] for key in corpus])  # Clustering con los sustantivos
    for n_clusters in range(1, 10):
        n_clusters += 1
        model_name = str(n_clusters) + "_" + reviews_type
        if not model_saved(model_path, model_name):
            peaks = retrieve_peaks(n_clusters, w2v_model, corpus)
            gmm = GaussianMixture(n_components=len(peaks), means_init=peaks).fit(embedding_corpus)
            clustering_results[model_name] = gmm
            save_gmm_model(os.path.join(model_path, model_name), gmm)
        else:
            gmm = load_gmm_model(os.path.join(model_path, model_name))
            clustering_results[model_name] = gmm
        aic_bic_results[model_name] = [gmm.aic(embedding_corpus), gmm.bic(embedding_corpus)]
        closest_idx, _ = pairwise_distances_argmin_min(gmm.means_, embedding_corpus)
        closest[model_name] = []
        for idx in closest_idx.tolist():
            closest[model_name].append(w2v_model.wv.index_to_key[idx])
    return clustering_results, aic_bic_results, closest


def retrieve_peaks(n_peaks, w2v_model, corpus):
    peaks = []
    last_index_found = 0
    for i in range(n_peaks):
        while last_index_found < len(w2v_model.wv.index_to_key):
            if w2v_model.wv.index_to_key[last_index_found] in corpus:
                peaks.append(w2v_model.wv[w2v_model.wv.index_to_key[last_index_found]])
                last_index_found += 1
                break
            last_index_found += 1
    return peaks


def retrieve_best_gmm_model(aic_bic_results):
    results_df = pd.DataFrame.from_dict(aic_bic_results, orient='index')
    results_df.columns = ["aic", "bic"]
    return results_df[results_df["aic"] == results_df["aic"].min()].index[0]


def retrieve_best_model_results(best_gmm_model_name, trained_models, w2v_model, nouns):
    labels = {}
    probabilities = {}
    n_clusters, target_model = best_gmm_model_name.split("_")
    model = trained_models[best_gmm_model_name]
    embedding_corpus = np.array([w2v_model.wv[key] for key in set(w2v_model.wv.index_to_key[:]).intersection(
        nouns)])  # Clustering con los sustantivos
    labels[target_model] = model.predict(embedding_corpus)
    probabilities[target_model] = model.score_samples(embedding_corpus)
    sample = np.array([key for key in set(w2v_model.wv.index_to_key[:]).intersection(set(nouns))])
    return probabilities, get_words_by_cluster(sample, target_model, labels, n_clusters), labels


def get_words_by_cluster(sample, target_model, labels, n_clusters):
    clusters = {}
    for id_cluster in range(int(n_clusters)):
        clusters[id_cluster] = np.array(list(sample))[np.where(labels[target_model] == id_cluster)[0]]
    return clusters


def perform_tsne(w2v_model, nouns, labels, figure_path):
    plt.figure(figsize=(15, 10))
    palette = sns.color_palette("bright", 10)
    tsne = TSNE(n_components=2, random_state=0)
    embedding_corpus = np.array([w2v_model.wv[key] for key in set(w2v_model.wv.index_to_key[:]).intersection(nouns)])
    X_embedded = tsne.fit_transform(X=embedding_corpus)
    ax = sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, legend='full',
                         palette=palette[:len(set(labels))])
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    # a = pd.concat({'x': pd.Series(X_embedded[:, 0]), 'y': pd.Series(X_embedded[:, 1]), 'val': pd.Series(np.array(set(w2v_model.wv.index_to_key[:]).intersection(nouns)))}, axis=1)
    # for i, point in a.iterrows():
    #     plt.gca().text(point['x']+.02, point['y'], str(point['val']))
    plt.savefig(os.path.join(figure_path, "topics.png"))


def save_topic_clusters_results(cluster_dict, results_path):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for key in cluster_dict:
        np.save(os.path.join(results_path, str(key) + '.npy'), cluster_dict[key])