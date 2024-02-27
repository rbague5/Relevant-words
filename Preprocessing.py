import time

from utils import logger
import os
import pickle
import re
import spacy
import logging
spacy.cli.download("es_core_news_sm")

import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
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
        if col in list(df.columns):
            logger.info(f"Dropping column {col} of the dataframe")
            df.drop(col, inplace=True, axis=1)


def load_data_and_save_as_csv(city, f):
    reviews = get_pickle(os.path.join(source_path, city), f + ".pkl")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    logger.info("Dropping columns")
    drop_columns(reviews, columns_to_delete[city])
    logger.info(f"Retrieving NOUNs from vocabulary, dataframe has got {len(reviews)} reviews.")
    start_time = time.time()
    # # Parallelize the processing using ThreadPoolExecutor
    # with ThreadPoolExecutor() as executor:
    #     reviews['nouns'] = list(executor.map(retrieve_noun_vocabulary, reviews['text']))
    reviews['nouns'] = reviews['text'].apply(lambda row: retrieve_noun_vocabulary(row))
    end_time = time.time()
    logger.info(f"NOUNs retrieved, in total took {int(end_time - start_time)} seconds, that is {int(end_time - start_time)/int(len(reviews))} seconds/review")
    logger.info("Preprocessing text")
    reviews['text'] = reviews['text'].apply(lambda row: preprocess_text(row))
    logger.info("Writting it to csv file")
    reviews.to_csv(os.path.join(target_path, city, f + '.csv'))
    return reviews


def load_items_and_save_as_csv(city, f):
    items = get_pickle(os.path.join(source_path, city), f + ".pkl")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    items.to_csv(os.path.join(target_path, city, f + '.csv'))
    return items


def load_users_and_save_as_csv(city, f):
    users = get_pickle(os.path.join(source_path, city), f + ".pkl")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    users.to_csv(os.path.join(target_path, city, f + '.csv'))
    return users


def preprocess_text(text):
    text = re.sub(r'[,!?;.-]', '', text)
    text = nltk.word_tokenize(text)
    text = [ch.lower() for ch in text if ch.isalpha()]
    return [word for word in text if not word in stopwords.words('spanish')]


def load_items_df(city, f):
    if os.path.exists(os.path.join(target_path, city, f + ".csv")):
        return pd.read_csv(os.path.join(target_path, city, f + ".csv"), encoding='utf-8')
    elif os.path.exists(os.path.join(source_path, city, f + ".pkl")):
        return load_items_and_save_as_csv(city, f)


def load_reviews_df(city, f):
    if os.path.exists(os.path.join(target_path, city, f + ".csv")):
        return pd.read_csv(os.path.join(target_path, city, f + ".csv"), encoding='utf-8')
    elif os.path.exists(os.path.join(source_path, city, f + ".pkl")):
        return load_data_and_save_as_csv(city, f)


def load_users_df(city, f):
    if os.path.exists(os.path.join(target_path, city, f + ".csv")):
        return pd.read_csv(os.path.join(target_path, city, f + ".csv"), encoding='utf-8')
    elif os.path.exists(os.path.join(source_path, city, f + ".pkl")):
        return load_users_and_save_as_csv(city, f)


def retrieve_noun_vocabulary(text):
    noun_tokens = set([])
    text = nlp(text)
    for token in text:
        if token.pos_ == 'NOUN':
            noun_tokens.add(str(token).lower())
    return list(noun_tokens)


def transform_target(df):
    if df['rating'] >= 30:
        return 1
    else:
        return 0
