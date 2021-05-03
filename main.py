import ast
import os

import utils
from Preprocessing import load_reviews_df

lower_limit_positive_rating = 30
upper_limit_negative_rating = 20
top_n_restaurants = 1
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
    most_commented_restaurants = data['restaurantId'].value_counts()
    for restaurant_id in most_commented_restaurants.head(1).index:
        topic_clustering(data, restaurant_id)
        # word_clustering()



def topic_clustering(data, restaurat_id):
    # Retrieve positive and negative reviews for each restaurantId
    restaurant_reviews = data[data['restaurantId'] == restaurat_id]
    positive_reviews_data = restaurant_reviews[restaurant_reviews['rating'] >= lower_limit_positive_rating]
    negative_reviews_data = restaurant_reviews[restaurant_reviews['rating'] <= upper_limit_negative_rating]

    corpus_positive = [ast.literal_eval(str_words) for str_words in positive_reviews_data['text'].values]
    corpus_negative = [ast.literal_eval(str_words) for str_words in negative_reviews_data['text'].values]

    corpus_nouns_positive = [ast.literal_eval(str_words) for str_words in positive_reviews_data['nouns'].values]
    corpus_nouns_positive = set([item for sublist in corpus_nouns_positive for item in sublist])
    corpus_nouns_negative = [ast.literal_eval(str_words) for str_words in negative_reviews_data['nouns'].values]
    corpus_nouns_negative = set([item for sublist in corpus_nouns_negative for item in sublist])

    #Entrenar tanto el modelo para comentario positivos como negativos (por ahora solo los positivos)
    for review_type in ['positive', 'negative']:
        corpus = corpus_positive if review_type == 'positive' else corpus_negative
        w2v_model = utils.train_w2v_model(os.path.join(w2v_models_path, str(restaurat_id)), review_type, corpus)
        nouns = set(corpus_nouns_positive) if review_type == "positive" else set(corpus_nouns_negative)
        trained_models, aic_bic_results, closest_words = utils.train_gmm_model(w2v_model, nouns, os.path.join(gmm_models_topics_path, review_type))
        best_gmm_model = utils.retrieve_best_gmm_model(aic_bic_results)
        probabilities, cluster_words, labels = utils.retrieve_best_model_results(best_gmm_model, trained_models, w2v_model, closest_words[best_gmm_model])

        utils.perform_tsne(w2v_model, labels, closest_words[best_gmm_model], os.path.join(figures_path, "topics", str(restaurat_id)), review_type)
        utils.save_topic_clusters_results(cluster_words, os.path.join(topics_clusters_path, review_type, str(restaurat_id)))
        # print(closest_words[best_gmm_model])
        # print(cluster_words)

if __name__ == "__main__":
    main()


#
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

