import os
import pickle
import re
import spacy
spacy.cli.download("es_core_news_sm")

import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')
nlp = spacy.load("es_core_news_sm")

columns_to_delete = {"gijon": ["images", "index", "language", "title", "url", "date"]}

base_path = "./dataset"
source_path = os.path.join(base_path, "raw")
target_path = os.path.join(base_path, "input")


def get_pickle(path, name):
    with open(os.path.join(path, name), 'rb') as handle:
        data = pickle.load(handle)
    return data


def drop_columns(df, columns):
    for col in columns:
        df.drop(col, inplace=True, axis=1)


def load_data_and_save_as_csv(city, f):
    reviews = get_pickle(os.path.join(source_path, city), f + ".pkl")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    drop_columns(reviews, columns_to_delete[city])
    reviews['nouns'] = reviews['text'].apply(lambda row: retrieve_noun_vocabulary(row))
    reviews['text'] = reviews['text'].apply(lambda row: preprocess_text(row))
    reviews.to_csv(os.path.join(target_path, city, f + '.csv'))
    return reviews


def preprocess_text(text):
    text = re.sub(r'[,!?;.-]', '', text)
    text = nltk.word_tokenize(text)
    text = [ch.lower() for ch in text if ch.isalpha()]
    return [word for word in text if not word in stopwords.words('spanish')]


def load_reviews_df(city, f):
    if os.path.exists(os.path.join(target_path, city, f + ".csv")):
        return pd.read_csv(os.path.join(target_path, city, f + ".csv"), encoding='utf-8')
    elif os.path.exists(os.path.join(source_path, city, f + ".pkl")):
        return load_data_and_save_as_csv(city, f)


def retrieve_noun_vocabulary(text):
    noun_tokens = set([])
    text = nlp(text)
    for token in text:
        if token.pos_ == 'NOUN':
            noun_tokens.add(str(token).lower())
    return list(noun_tokens)


def transform_target(df):
    if df['rating'] >= 40:
        return 1
    else:
        return 0
