from itertools import chain

import utils
from config import *
from utils import logger
import pandas as pd
import ast
import os
from nltk.corpus import words
from preprocessing import load_reviews_df, load_items_df, load_users_df


def main_analysis_by_city(data, city):
    logger.info(f"Doing topic clustering for city: {city}")
    topic_clustering(data, city)


def main_analysis_by_restaurant(data, city):
    most_commented_restaurants = data['itemId'].value_counts()
    for restaurant_id in most_commented_restaurants.head(top_n_restaurants).index:
        logger.info(f"Doing topic clustering for city: {city} and restaurant id: {restaurant_id}")
        topic_clustering(data, city, restaurant_id)


def main_analysis_by_restaurant_tf_itf(data, city):
    path = os.path.join("results_tf_itf", city)
    # tf_results = utils.retrieve_word_frequencies(utils.flatten_reviews_by_restaurant(data)['nouns'])
    tf_positive, tf_negative = utils.retrieve_word_frequencies_by_review(data)
    tf_by_restaurant = pd.concat([tf_positive, tf_negative], ignore_index=True)
    most_commented_restaurants = data['itemId'].value_counts()

    restaurant_ids = most_commented_restaurants.head(top_n_restaurants).index
    y_max = tf_by_restaurant[tf_by_restaurant['itemId'].isin(restaurant_ids)]['frequency'].max()
    y_max = utils.roundup(y_max)
    for restaurant_id in most_commented_restaurants.head(top_n_restaurants).index:
        utils.generate_histogram(tf_by_restaurant[tf_by_restaurant['itemId'] == restaurant_id], path, 20, y_max)
        utils.generate_histogram(tf_positive[tf_positive['itemId'] == restaurant_id], path, 20, y_max,"positive")
        utils.generate_histogram(tf_negative[tf_negative['itemId'] == restaurant_id], path, 20, y_max,"negative")


def get_corpus(data, threshold, restaurant_id=None):

    if restaurant_id is not None:
        logger.info(f"Filtering data for restaurant_id: {restaurant_id}")
        data = data[data['itemId'] == restaurant_id]

    positive_reviews_data = data[data['rating'] >= threshold]
    negative_reviews_data = data[data['rating'] < threshold]
    logger.info(f"Total reviews {len(data.index)}, positive: {len(positive_reviews_data.index)}, negative: {len(negative_reviews_data.index)}")

    load_nltk()
    english_words = set(words.words())

    corpus_positive = [
        str_words if isinstance(str_words, list) else ast.literal_eval(str_words)
        for str_words in positive_reviews_data['text'].values
    ]
    corpus_positive = list(set(chain.from_iterable(corpus_positive)))

    corpus_nouns_positive = [
        str_words if isinstance(str_words, list) else ast.literal_eval(str_words)
        for str_words in positive_reviews_data['nouns'].values
    ]
    corpus_nouns_positive = [[word.strip() for word in set(nouns) if word.lower() in english_words] for nouns in corpus_nouns_positive]
    corpus_nouns_positive = list(set(chain.from_iterable(corpus_nouns_positive)))

    corpus_negative = [
        str_words if isinstance(str_words, list) else ast.literal_eval(str_words)
        for str_words in negative_reviews_data['text'].values
    ]
    corpus_negative = list(set(chain.from_iterable(corpus_negative)))

    corpus_nouns_negative = [
        str_words if isinstance(str_words, list) else ast.literal_eval(str_words)
        for str_words in negative_reviews_data['nouns'].values
    ]
    corpus_nouns_negative = [[word.strip() for word in set(nouns) if word.lower() in english_words] for nouns in corpus_nouns_negative]
    corpus_nouns_negative = list(set(chain.from_iterable(corpus_nouns_negative)))

    return corpus_positive, corpus_nouns_positive, corpus_negative, corpus_nouns_negative


def topic_clustering(data, city, restaurant_id=None):
    corpus_positive, corpus_nouns_positive, corpus_negative, corpus_nouns_negative = get_corpus(data, rating_threshold, restaurant_id)

    covariance_types = ['full', 'diag']
    reg_covars = [1e-6, 1e-5, 1e-4]
    max_iters = [100, 200]
    n_inits = [1, 3]

    params_list = utils.generate_grid_search_params(covariance_types, reg_covars, max_iters, n_inits)

    metric_directions = {
        'silhouette_score': 1,
        'intra_distance': -1,
        'inter_distance': 1,
        'semantic_coherence': 1,
        'npmi': 1,
        'scci': 1,
    }

    for class_review in ['positive', 'negative']:
        all_clusters_metrics = {}
        logger.info(f"Calculating {class_review} with model: {embedding_model_name} for city: {city}")
        # corpus = corpus_positive if class_review == 'positive' else corpus_negative
        embedding_model = utils.train_or_load_embedding_model(model_name=embedding_model_name)
        nouns = set(corpus_nouns_positive) if class_review == "positive" else set(corpus_nouns_negative)

        for params in params_list:
            logger.info(f"Training GMM model with params: {params}")
            img_name = "_".join([f"{k}-{v}" for k, v in params.items()]).replace(".", "_").replace("covariance", "cov")

            if restaurant_id is not None:
                city_path = os.path.join("results_by_restaurant", str(city), str(restaurant_id))
                embedding_model_path = os.path.join(city_path, img_name, w2v_models_path_by_city, embedding_model_name)
                gmm_path = os.path.join(city_path, img_name, gmm_models_path_by_city, class_review)
                topics_path_images = os.path.join(city_path, img_name, figures_path_by_city)
                topics_path_nouns = os.path.join(city_path, img_name, topics_clusters_path_by_city)
                cluster_metrics_path = os.path.join(city_path, img_name, "metrics", class_review)
            else:
                city_path = os.path.join("results_by_city", str(city))
                embedding_model_path = os.path.join(city_path, img_name, w2v_models_path_by_city, embedding_model_name)
                gmm_path = os.path.join(city_path, img_name, gmm_models_path_by_city, class_review)
                topics_path_images = os.path.join(city_path, img_name, figures_path_by_city)
                topics_path_nouns = os.path.join(city_path, img_name, topics_clusters_path_by_city)
                cluster_metrics_path = os.path.join(city_path, img_name, "metrics", class_review)

            # Train or load GMM model
            trained_models, cluster_metrics, closest_words = utils.train_gmm_model(
                embedding_model, nouns, gmm_path, n_clusters_range=n_clusters_range, top_n_points=top_n_nearest_points,
                covariance_type=params["covariance_type"], reg_covar=params["reg_covar"],
                max_iter=params["max_iter"], n_init=params["n_init"]
            )

            utils.save_cluster_metrics(cluster_metrics, file_path=os.path.join(cluster_metrics_path, f"all_metrics"))
            for k, v in cluster_metrics.items():
                if "scci" in v and isinstance(v["scci"], dict):
                    v["scci"] = v["scci"]["scci"]
            best_clusters = utils.select_best_clusters(cluster_metrics)
            all_clusters_metrics[img_name] = best_clusters
            utils.save_cluster_metrics(best_clusters, file_path=os.path.join(cluster_metrics_path, f"best_metrics"))

            for metric, best_gmm_model_metric in best_clusters.items():
                if best_gmm_model_metric is not None:
                    logger.info(f"Best cluster for {metric}: {best_gmm_model_metric}")
                    best_gmm_model = list(best_gmm_model_metric.keys())[0]
                    probabilities, cluster_words, labels = utils.retrieve_best_model_results(best_gmm_model, trained_models, embedding_model, closest_words[best_gmm_model])

                    logger.info(f"Performing TSNE")
                    utils.perform_tsne(embedding_model, labels, closest_words[best_gmm_model], topics_path_images, class_review, metric)

                    logger.info(f"Saving topic clusters results")
                    utils.save_topic_clusters_results(cluster_words, topics_path_nouns, class_review, metric)

                    logger.info(f"Closest words {closest_words[best_gmm_model]}")
                    logger.info(f"Cluster words: {cluster_words}")

        best_per_metric = {}
        for metric in metric_directions:
            best_config = None
            best_value = None
            direction = metric_directions[metric]

            for config, metrics in all_clusters_metrics.items():
                value_dict = metrics.get(metric)
                if value_dict is None:
                    continue

                num_clusters = list(value_dict.keys())[0]
                value = list(value_dict.values())[0]

                if best_value is None:
                    best_value = value
                    best_config = config
                else:
                    if (direction == 1 and value > best_value) or (direction == -1 and value < best_value):
                        best_value = value
                        best_config = config

            best_per_metric[metric] = str((best_config, num_clusters, best_value))

        utils.save_cluster_metrics(best_per_metric, file_path=os.path.join(city_path, f"best_overall_{class_review}"))


if __name__ == "__main__":
    # ["gijon", "moscow", "madrid", "istanbul", "barcelona"]
    for city in ["moscow", "madrid", "istanbul", "barcelona"]:
        logger.info(f"Checking city: {city.title()}")

        logger.info(f"Creating reviews data")
        _ = load_items_df(city, "items")
        _ = load_users_df(city, "users")
        data = load_reviews_df(city, "reviews", lang="en", from_date="2018-01-01", to_date="2023-01-01")

        logger.info(f"Data of {city} loaded, in total there are {len(data.index)} reviews")
        logger.info(f"N. of users: {data['userId'].nunique()}")
        logger.info(f"N. of restaurants: {data['itemId'].nunique()}")
        logger.info(f"N. of reviews: {data['reviewId'].nunique()}")
        logger.info(f"N. of rating: {data['rating'].value_counts().sort_index(ascending=False)}")
        logger.info(f"Review dates between: {data['date'].min()} and {data['date'].max()}")

        main_analysis_by_city(data, city)
        # main_analysis_by_restaurant(data, city)
        # main_analysis_by_restaurant_tf_itf(data, city)



# #############################
# ##### ONE HOT ENCODING ######
# #############################
# '''
# Se recuperan únicamente aquellas palabras que aparecen al menos un 10% de las veces (faltas de ortografías, emojis, etc...)
# cv.vocabulary_: {"word": idx_vocabulary}
# corpus_enc: {documento: {idx_vocabulary: # ocurrencias en cada documento }}
# '''
#
# ##### POSITIVE REVIEWS ######
# cv_pos = CountVectorizer(min_df=0.01)
# corpus_positive_enc = cv_pos.fit_transform(corpus_positive)
#
# binarizer = Binarizer()
# corpus_positive_one_hot_enc = binarizer.fit_transform(corpus_positive_enc.toarray())
#
# corpus_positive = pd.DataFrame(corpus_positive_one_hot_enc)
# colnames = ["w"+str(i) for i in range(len(cv_pos.vocabulary_))]
# corpus_positive.columns = colnames
#
# print("Número total de comentarios positivos:{}".format(len(corpus_positive)))
# print("Tamaño del vocabulario:{}", len(cv_pos.vocabulary_))
#
#
# ##### NEGATIVE REVIEWS ######
# cv_neg = CountVectorizer(min_df=0.01)
# corpus_negative_enc = cv_neg.fit_transform(corpus_negative)
#
# binarizer = Binarizer()
# corpus_negative_one_hot_enc = binarizer.fit_transform(corpus_negative_enc.toarray())
#
# corpus_negative = pd.DataFrame(corpus_negative_one_hot_enc)
# colnames = ["w"+str(i) for i in range(len(cv_neg.vocabulary_))]
# corpus_negative.columns = colnames
# print("Número total de comentarios negative:{}".format(len(corpus_negative)))
# print("Tamaño del vocabulario:{}", len(cv_neg.vocabulary_))
#
#
# k_results = {}
#
# for idx, value in enumerate(F_mim):
#     clusters = idx+1
#     centroids = np.zeros((clusters, len(cv.vocabulary_)), dtype=np.int)
#     centroids[list(range(clusters)), F_mim[:clusters, np.newaxis]] = 1
#
#     positive_reviews_enc = data_enc[data_enc[data_enc.columns[-1]] == 1]
#     negative_reviews_enc = data_enc[data_enc[data_enc.columns[-1]] == 0]
#
#     gm_positive = GaussianMixture(n_components=clusters, means_init=centroids).fit(positive_reviews_enc[positive_reviews_enc.columns[:-1]])
#     k_results[str(clusters)+"_positive"] = gm_positive
#     gm_negative = GaussianMixture(n_components=clusters, means_init=centroids).fit(positive_reviews_enc[negative_reviews_enc.columns[:-1]])
#     k_results[str(clusters)+"_negative"] = gm_negative


