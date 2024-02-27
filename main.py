import sys

from utils import logger
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from gensim.models import Word2Vec
import ast
import utils
import os
from sklearn.metrics import pairwise_distances_argmin_min
from Preprocessing import load_reviews_df, load_items_df, load_users_df

lower_limit_positive_rating = 30 #TODO cambiar a 30 el límite inferior de los comentarios positivos
upper_limit_negative_rating = 20
top_n_restaurants = 10
w2v_models_path = "./models/word2vec"
gmm_models_path = "./models/gmm"
topics_clusters_path = "./results/topics"
figures_path = "./results/figure"
positive_model_path = os.path.join(w2v_models_path, "positive")
negative_model_path = os.path.join(w2v_models_path, "negative")


def main():
    city = "gijon"
    logger.info(f"Checking city: {city}")
    logger.info(f"Creating items data")
    data = load_items_df("gijon", "items")
    logger.info(f"Creating users data")
    data = load_users_df("gijon", "users")
    logger.info(f"Creating reviews data")
    data = load_reviews_df("gijon", "reviews")
    logger.info(f"Data of {city} loaded, in total there are {len(data.index)} reviews")
    logger.info(f"Doing topic clustering")
    topic_clustering(data, top_n_restaurants)


def topic_clustering(data, num_restaurants):
    most_commented_restaurants = data['itemId'].value_counts()
    logger.info(f"Most commented restaurants: {most_commented_restaurants}")
    logger.info(most_commented_restaurants.head(top_n_restaurants).index)
    for restaurantId in most_commented_restaurants.head(num_restaurants).index:
        logger.info(f"Checking restaurant {restaurantId}")
        # Retrieve positive and negative reviews for each restaurantId
        restaurant_reviews = data[data['itemId'] == restaurantId]
        positive_reviews_data = data[data['rating'] >= lower_limit_positive_rating]
        negative_reviews_data = data[data['rating'] <= upper_limit_negative_rating]
        logger.info(f"{restaurantId}: Total reviews {len(restaurant_reviews.index)},"
                    f"positive: {len(positive_reviews_data.index)}, negative: {len(negative_reviews_data.index)}")

        corpus_positive = [ast.literal_eval(str_words) for str_words in positive_reviews_data['text'].values]
        corpus_negative = [ast.literal_eval(str_words) for str_words in negative_reviews_data['text'].values]

        corpus_nouns_positive = [ast.literal_eval(str_words) for str_words in positive_reviews_data['nouns'].values]
        corpus_nouns_positive = set([item for sublist in corpus_nouns_positive for item in sublist])
        corpus_nouns_negative = [ast.literal_eval(str_words) for str_words in negative_reviews_data['nouns'].values]
        corpus_nouns_negative = set([item for sublist in corpus_nouns_negative for item in sublist])

        # Entrenar tanto el modelo para comentario positivos como negativos (por ahora solo los positivos)
        for class_review in ['positive', 'negative']:
            logger.info(f"Training w2v {class_review} for restaurantId: {restaurantId}")
            w2v_model = utils.train_w2v_model(w2v_models_path, class_review, corpus_positive)
            nouns = set(corpus_nouns_positive) if class_review == "positive" else set(corpus_nouns_negative)
            corpus = corpus_positive if class_review == "positive" else corpus_negative
            logger.info(f"Training gmm {class_review} for restaurantId: {restaurantId}")
            trained_models, aic_bic_results, closest_words = utils.train_gmm_model(w2v_model, class_review, nouns, gmm_models_path)
            best_gmm_model = utils.retrieve_best_gmm_model(aic_bic_results)
            probabilities, cluster_words, labels = utils.retrieve_best_model_results(best_gmm_model, trained_models, w2v_model, nouns)
            logger.info(f"Performing TSNE")
            utils.perform_tsne(w2v_model, nouns, labels[class_review], os.path.join(figures_path, str(restaurantId), class_review))
            logger.info(f"Saving topic clusters results")
            utils.save_topic_clusters_results(cluster_words, os.path.join(topics_clusters_path, str(restaurantId), class_review))
            logger.info(f"Closest words {closest_words[best_gmm_model]}")
            logger.info(f"Cluster words: {cluster_words}")


if __name__ == "__main__":
    main()



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


