import utils
from config import *
from utils import logger
import pandas as pd
import ast
import os

from preprocessing import load_reviews_df, load_items_df, load_users_df


def main_analysis_by_city(data, city):
    logger.info(f"Doing topic clustering")
    topic_clustering_by_city(data, city)


def main_analysis_by_restaurant(data):
    most_commented_restaurants = data['itemId'].value_counts()
    for restaurant_id in most_commented_restaurants.head(top_n_restaurants).index:
        logger.info(f"Doing topic clustering for restaurant id: {restaurant_id}")
        topic_clustering_by_restaurant(data, restaurant_id)


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


def get_corpus(data, threshold):
    positive_reviews_data = data[data['rating'] >= threshold]
    negative_reviews_data = data[data['rating'] < threshold]
    logger.info(f"Total reviews {len(data.index)}, positive: {len(positive_reviews_data.index)}, negative: {len(negative_reviews_data.index)}")

    corpus_positive = [ast.literal_eval(str_words) for str_words in positive_reviews_data['text'].values]
    corpus_positive = [list(set(review_positive)) for review_positive in corpus_positive]
    corpus_nouns_positive = [ast.literal_eval(str_words) for str_words in positive_reviews_data['nouns'].values]
    corpus_nouns_positive = [list(set(nouns_review_positive)) for nouns_review_positive in corpus_nouns_positive]

    corpus_negative = [ast.literal_eval(str_words) for str_words in negative_reviews_data['text'].values]
    corpus_negative = [list(set(review_negative)) for review_negative in corpus_negative]
    corpus_nouns_negative = [ast.literal_eval(str_words) for str_words in negative_reviews_data['nouns'].values]
    corpus_nouns_negative = [list(set(nouns_review_negative)) for nouns_review_negative in corpus_nouns_negative]

    return corpus_positive, corpus_nouns_positive, corpus_negative, corpus_nouns_negative


def topic_clustering_by_city(data, city):
    corpus_positive, corpus_nouns_positive, corpus_negative, corpus_nouns_negative = get_corpus(data, rating_threshold)
    # Entrenar tanto el modelo para comentario positivos como negativos
    for class_review in ['positive', 'negative']:

        w2v_path = os.path.join("results_by_city", city, w2v_models_path_by_city)
        gmm_path = os.path.join("results_by_city", city, gmm_models_path_by_city, class_review)
        topics_path_images = os.path.join("results_by_city", city, figures_path_by_city)
        topics_path_nouns = os.path.join("results_by_city", str(city), topics_clusters_path_by_city, class_review)
        cluster_metrics_path = os.path.join("results_by_city", str(city), "metrics", class_review)

        logger.info(f"Training w2v {class_review} for city: {city}")
        corpus = corpus_positive if class_review == 'positive' else corpus_negative
        w2v_model = utils.train_w2v_model(w2v_path, class_review, corpus)
        nouns = set(corpus_nouns_positive) if class_review == "positive" else set(corpus_nouns_negative)
        trained_models, cluster_metrics, closest_words = utils.train_gmm_model(w2v_model, nouns, gmm_path, n_clusters_range=range(2, 15))
        utils.save_cluster_metrics(cluster_metrics, file_path=os.path.join(cluster_metrics_path, "all"))
        # best_clusters = utils.select_best_clusters(cluster_metrics)
        utils.save_cluster_metrics(cluster_metrics, file_path=os.path.join(cluster_metrics_path, "best"))
        weights = {"semantic_coherence": 2, "silhouette_score": 1}  # Prioritize semantic quality
        thresholds = {"semantic_coherence": 0.1, "silhouette_score": 0.1}
        # filtered_clusters = utils.filter_clusters(cluster_metrics, thresholds)
        best_gmm_model = utils.select_overall_best_cluster(cluster_metrics, weights)
        # probabilities, cluster_words, labels = utils.retrieve_best_model_results(best_gmm_model, trained_models, w2v_model, closest_words[best_gmm_model])

        metric = f'semantic_coherence-{weights["semantic_coherence"]}_silhouette_score-{weights["silhouette_score"]}'

        # for metric, best_gmm_model in best_clusters.items():
        # logger.info(f"Best cluster for {metric}: {best_gmm_model} with score {cluster_metrics[best_gmm_model][metric]}")

        probabilities, cluster_words, labels = utils.retrieve_best_model_results(best_gmm_model, trained_models, w2v_model, closest_words[best_gmm_model])

        logger.info(f"Performing TSNE")
        utils.perform_tsne(w2v_model, labels, closest_words[best_gmm_model], os.path.join(topics_path_images, metric), class_review)

        logger.info(f"Saving topic clusters results")
        utils.save_topic_clusters_results(cluster_words, os.path.join(topics_path_nouns, metric))

        logger.info(f"Closest words {closest_words[best_gmm_model]}")
        logger.info(f"Cluster words: {cluster_words}")


def topic_clustering_by_restaurant(data, restaurat_id):
    # Retrieve positive and negative reviews for each restaurantId
    restaurant_reviews = data[data['itemId'] == restaurat_id]
    logger.info(f"Analysing restaurant: {restaurat_id}")
    corpus_positive, corpus_nouns_positive, corpus_negative, corpus_nouns_negative = get_corpus(restaurant_reviews, rating_threshold)

    # Entrenar tanto el modelo para comentario positivos como negativos (por ahora solo los positivos)
    for review_type in ['positive', 'negative']:
        w2v_path = os.path.join("results_by_restaurant", city, w2v_models_path_by_restaurant)
        gmm_path = os.path.join("results_by_restaurant", city, gmm_models_path_by_restaurant, review_type)
        topics_path_images = os.path.join("results_by_restaurant", city, figures_path_by_restaurant)
        topics_path_nouns = os.path.join("results_by_restaurant", city, topics_clusters_path_by_restaurant, review_type)
        cluster_metrics_path = os.path.join("results_by_restaurant", str(city), "metrics", review_type)

        logger.info(f"Training w2v {review_type} for restaurant: {restaurat_id}")

        corpus = corpus_positive if review_type == 'positive' else corpus_negative
        w2v_model = utils.train_w2v_model(os.path.join(w2v_path, review_type, str(restaurat_id)), review_type, corpus)
        nouns = set(corpus_nouns_positive) if review_type == "positive" else set(corpus_nouns_negative)
        # trained_models, aic_bic_results, closest_words, best_gmm_model = utils.train_gmm_model(w2v_model, nouns, os.path.join(gmm_path, str(restaurat_id)), n_clusters_range=range(2, 15))
        # # best_gmm_model = utils_analisys_by_restaurant.retrieve_best_gmm_model(aic_bic_results)
        # probabilities, cluster_words, labels = utils.retrieve_best_model_results(best_gmm_model, trained_models, w2v_model, closest_words[best_gmm_model])

        trained_models, cluster_metrics, closest_words = utils.train_gmm_model(w2v_model, nouns, os.path.join(gmm_path, str(restaurat_id)), n_clusters_range=range(2, 15))
        utils.save_cluster_metrics(cluster_metrics, file_path=os.path.join(cluster_metrics_path, str(restaurat_id)))
        # best_gmm_model = utils_analisys_by_city.retrieve_best_gmm_model(aic_bic_results)
        # best_clusters = utils.select_best_clusters(cluster_metrics)
        weights = {"semantic_coherence": 2, "silhouette_score": 1}  # Prioritize semantic quality
        thresholds = {"semantic_coherence": 0.1, "silhouette_score": 0.1}
        # filtered_clusters = utils.filter_clusters(cluster_metrics, thresholds)
        best_gmm_model = utils.select_overall_best_cluster(cluster_metrics, weights)

        probabilities, cluster_words, labels = utils.retrieve_best_model_results(best_gmm_model, trained_models, w2v_model, closest_words[best_gmm_model])
        metric = f'semantic_coherence-{weights["semantic_coherence"]}_silhouette_score-{weights["silhouette_score"]}'
        logger.info("Performing TSNE")
        utils.perform_tsne(w2v_model, labels, closest_words[best_gmm_model], os.path.join(topics_path_images, str(restaurat_id), metric), review_type)
        logger.info("Saving topic clusters results")
        utils.save_topic_clusters_results(cluster_words, os.path.join(topics_path_nouns, str(restaurat_id), metric))
        logger.info(f"Closest words {closest_words[best_gmm_model]}")
        logger.info(f"Cluster words: {cluster_words}")

        # for metric, best_gmm_model in best_cluster.items():
        #     logger.info(f"Best cluster for {metric}: {best_gmm_model} with score {cluster_metrics[best_gmm_model][metric]}")
        #
        #     probabilities, cluster_words, labels = utils.retrieve_best_model_results(best_gmm_model, trained_models, w2v_model, closest_words[best_gmm_model])
        #
        #     logger.info("Performing TSNE")
        #     utils.perform_tsne(w2v_model, labels, closest_words[best_gmm_model], os.path.join(topics_path_images, str(restaurat_id), metric), review_type)
        #     logger.info("Saving topic clusters results")
        #     utils.save_topic_clusters_results(cluster_words, os.path.join(topics_path_nouns, str(restaurat_id), metric))
        #     logger.info(f"Closest words {closest_words[best_gmm_model]}")
        #     logger.info(f"Cluster words: {cluster_words}")


if __name__ == "__main__":
    # ["gijon", "moscow", "madrid", "Istanbul", "barcelona"]
    for city in ["moscow"]:
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
        main_analysis_by_restaurant(data)
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


