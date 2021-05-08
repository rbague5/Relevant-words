import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import Binarizer
from gensim.models import Word2Vec
import ast
import utils
import os
import scipy.stats
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

from Preprocessing import load_reviews_df

lower_limit_positive_rating = 30
upper_limit_negative_rating = 20
top_n_restaurants = 5
w2v_models_path = "./models/word2vec"
gmm_models_topics_path = "./models/gmm/topics"
gmm_models_words_path = "./models/gmm/words"
topics_clusters_path = "./results/topics"
words_clusters_path = "./results/words"
figures_path = "./results/figure"
positive_model_path = os.path.join(w2v_models_path, "positive")
negative_model_path = os.path.join(w2v_models_path, "negative")


def main():
    data = load_reviews_df("gijon", "reviews")
    # tf_results = utils.retrieve_word_frequencies(utils.flatten_reviews_by_restaurant(data)['nouns'])
    tf_positive, tf_negative = utils.retrieve_word_frequencies_by_review(data)
    tf_by_restaurant =  pd.concat([tf_positive, tf_negative], ignore_index=True)
    most_commented_restaurants = data['restaurantId'].value_counts()

    restaurant_ids = most_commented_restaurants.head(top_n_restaurants).index
    y_max = tf_by_restaurant[tf_by_restaurant['restaurantId'].isin(restaurant_ids)]['frequency'].max()
    y_max = utils.roundup(y_max)
    for restaurant_id in most_commented_restaurants.head(top_n_restaurants).index:
        utils.generate_histogram(tf_by_restaurant[tf_by_restaurant['restaurantId'] == restaurant_id], figures_path, 20, y_max)
        utils.generate_histogram(tf_positive[tf_positive['restaurantId'] == restaurant_id], figures_path, 20, y_max, "positive")
        utils.generate_histogram(tf_negative[tf_negative['restaurantId'] == restaurant_id], figures_path, 20, y_max, "negative")


if __name__ == "__main__":
    main()

#############################
##### ONE HOT ENCODING ######
#############################
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
#
#
# # k_results = {}
# #
# # for idx, value in enumerate(F_mim):
# #     clusters = idx+1
# #     centroids = np.zeros((clusters, len(cv.vocabulary_)), dtype=np.int)
# #     centroids[list(range(clusters)), F_mim[:clusters, np.newaxis]] = 1
# #
# #     positive_reviews_enc = data_enc[data_enc[data_enc.columns[-1]] == 1]
# #     negative_reviews_enc = data_enc[data_enc[data_enc.columns[-1]] == 0]
# #
# #     gm_positive = GaussianMixture(n_components=clusters, means_init=centroids).fit(positive_reviews_enc[positive_reviews_enc.columns[:-1]])
# #     k_results[str(clusters)+"_positive"] = gm_positive
# #     gm_negative = GaussianMixture(n_components=clusters, means_init=centroids).fit(positive_reviews_enc[negative_reviews_enc.columns[:-1]])
# #     k_results[str(clusters)+"_negative"] = gm_negative
#


# import functools
# import time
#
# def timer(func):
#     @functools.wraps(func)
#     def wrapper_timer(*args, **kwargs):
#         tic = time.perf_counter()
#         value = func(*args, **kwargs)
#         toc = time.perf_counter()
#         elapsed_time = toc - tic
#         print(f"Elapsed time: {elapsed_time:0.4f} seconds")
#         return value
#     return wrapper_timer