import ast
import os

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
import math

sns.set(rc={'figure.figsize': (15, 10)})
sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})
sns.set(font_scale=0.8)


def retrieve_word_frequencies(data):
    cv = CountVectorizer(min_df=0.10)
    dict_results = {"restaurantId": [], "word": [], "frequency": []}
    vectorized_results = cv.fit_transform(data)
    for restaurant_idx, results in enumerate(vectorized_results):
        freq = results.data
        idx_vocabulary = results.indices
        for i, j in zip(freq, idx_vocabulary):
            dict_results["restaurantId"].append(data.index[restaurant_idx])
            dict_results["word"].append(list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(j)])
            dict_results["frequency"].append(i)
    res = pd.DataFrame().from_dict(dict_results)
    res.sort_values(['restaurantId', 'frequency'], ascending=[True, False], inplace=True)
    return res


def retrieve_word_frequencies_by_review(data):
    tf_positive = retrieve_word_frequencies(flatten_reviews_by_restaurant(data[data['rating'] >= 30])['nouns'])
    tf_negative = retrieve_word_frequencies(flatten_reviews_by_restaurant(data[data['rating'] < 30])['nouns'])
    return tf_positive, tf_negative


def flatten_reviews_by_restaurant(data):
    data_grouped_by_restaurant = data.groupby('restaurantId').agg(
        {'nouns': lambda x: [' '.join(ast.literal_eval(w)) for w in list((np.hstack(x)))]})
    data_grouped_by_restaurant['nouns'] = data_grouped_by_restaurant['nouns'].apply(lambda x: ' '.join(x))
    return data_grouped_by_restaurant


def generate_histogram(df, figure_path, top_n, y_max, type_review=None):
    plt.figure(figsize=(15, 10))
    pallete = "Blues_r"
    if type_review is not None:
        if type_review == "positive":
            pallete = "BuGn_r"
        elif type_review == "negative":
            pallete = "OrRd_r"
    ax = sns.barplot(x="word", y="frequency", data=df[:top_n],
                     palette=pallete)
    ax.set_ylabel('count')
    ax.set_ylim([0, y_max])
    path = os.path.join(figure_path, str(df['restaurantId'].values[0]))
    if not os.path.exists(path):
        os.makedirs(path)
    filename = type_review if type_review is not None else 'all_reviews'
    plt.savefig(os.path.join(path, filename+".png"))

    # https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92


def roundup(x):
    return 100 + int(math.ceil(x / 100.0)) * 100