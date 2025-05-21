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

    for class_review in ['positive', 'negative']:
        if restaurant_id is not None:
            embedding_model_path = os.path.join("results_by_restaurant", str(city), str(restaurant_id), w2v_models_path_by_city, embedding_model_name)
            gmm_path = os.path.join("results_by_restaurant", str(city), str(restaurant_id), gmm_models_path_by_city, class_review)
            topics_path_images = os.path.join("results_by_restaurant", str(city), str(restaurant_id), figures_path_by_city)
            topics_path_nouns = os.path.join("results_by_restaurant", str(city), str(restaurant_id), topics_clusters_path_by_city)
            cluster_metrics_path = os.path.join("results_by_restaurant", str(city), str(restaurant_id), "metrics", class_review)
        else:
            embedding_model_path = os.path.join("results_by_city", city, w2v_models_path_by_city, embedding_model_name)
            gmm_path = os.path.join("results_by_city", city, gmm_models_path_by_city, class_review)
            topics_path_images = os.path.join("results_by_city", city, figures_path_by_city)
            topics_path_nouns = os.path.join("results_by_city", str(city), topics_clusters_path_by_city)
            cluster_metrics_path = os.path.join("results_by_city", str(city), "metrics", class_review)

        logger.info(f"Calculating {class_review} with model: {embedding_model_name} for city: {city}")
        # corpus = corpus_positive if class_review == 'positive' else corpus_negative
        embedding_model = utils.train_or_load_embedding_model(model_name=embedding_model_name)
        nouns = set(corpus_nouns_positive) if class_review == "positive" else set(corpus_nouns_negative)
        trained_models, cluster_metrics, closest_words = utils.train_gmm_model(embedding_model, nouns, gmm_path, n_clusters_range=n_clusters_range, top_n_points=top_n_nearest_points)
        utils.save_cluster_metrics(cluster_metrics, file_path=os.path.join(cluster_metrics_path, "all"))
        best_clusters = utils.select_best_clusters(cluster_metrics)
        utils.save_cluster_metrics(best_clusters, file_path=os.path.join(cluster_metrics_path, "best"))

        for metric, best_gmm_model in best_clusters.items():
            if best_gmm_model is not None:
                logger.info(f"Best cluster for {metric}: {best_gmm_model} with score {cluster_metrics[best_gmm_model][metric]}")
                probabilities, cluster_words, labels = utils.retrieve_best_model_results(best_gmm_model, trained_models, embedding_model, closest_words[best_gmm_model])

                logger.info(f"Performing TSNE")
                utils.perform_tsne(embedding_model, labels, closest_words[best_gmm_model], topics_path_images, class_review, metric)

                logger.info(f"Saving topic clusters results")
                utils.save_topic_clusters_results(cluster_words, topics_path_nouns, class_review, metric)

                logger.info(f"Closest words {closest_words[best_gmm_model]}")
                logger.info(f"Cluster words: {cluster_words}")

if __name__ == "__main__":
    # ["gijon", "moscow", "madrid", "istanbul", "barcelona"]
    for city in ["barcelona"]:
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

        # main_analysis_by_city(data, city)
        main_analysis_by_restaurant(data, city)
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


