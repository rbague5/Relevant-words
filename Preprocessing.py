import time
from datetime import timedelta
import pandas as pd

from config import target_path, source_path, load_spacy
from utils import logger
import os
import pickle
import re
import nltk
from nltk.corpus import stopwords


def get_pickle(path, name):
    with open(os.path.join(path, name), 'rb') as handle:
        data = pickle.load(handle)
    return data


def drop_columns(df, columns):
    for col in columns:
        if col in list(df.columns):
            logger.info(f"Dropping column {col} of the dataframe")
            df.drop(col, inplace=True, axis=1)


def load_pkl_and_save_as_csv(city, f):
    pkl = get_pickle(os.path.join(source_path, city), f + ".pkl")
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if not os.path.exists(os.path.join(target_path, city)):
        os.makedirs(os.path.join(target_path, city))
    pkl.to_csv(os.path.join(target_path, city, f + '.csv'))
    return pkl


def clean_non_english(review):
    # Regex to check for Korean (Hangul), Chinese (Han), Cyrillic, and Hebrew characters
    pattern = re.compile(r'[\uAC00-\uD7A3\u4E00-\u9FFF\u0400-\u04FF\u0590-\u05FF]')
    return bool(pattern.search(review))


def load_data_and_save_as_csv(city, f, lang, from_date="2018-01-01", to_date="2023-01-01"):
    # Load reviews from a pickle file
    reviews = get_pickle(os.path.join(source_path, city), f + ".pkl")
    reviews = reviews[reviews['language'] == lang]
    reviews['date'] = pd.to_datetime(reviews['date'])


    reviews = reviews[(reviews['date'] >= from_date) & (reviews['date'] < to_date)]

    # Filter out reviews without English chars
    mask = ~reviews['text'].apply(clean_non_english)
    reviews = reviews[mask]

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    nlp = load_spacy()
    reviews = reviews.drop_duplicates(subset=['reviewId']).reset_index(drop=True)
    # Preprocess text in parallel
    start_time = time.time()
    logger.info("Preprocessing text")
    reviews['text'] = reviews['text'].apply(lambda row: preprocess_text(row, nlp))
    total_seconds = round(time.time() - start_time, 2)
    logger.info(f"Text preprocessed, took {str(timedelta(seconds=int(total_seconds)))} sec -> {total_seconds / len(reviews):.4f} sec/review")

    # Retrieve nouns in parallel
    start_time = time.time()
    logger.info(f"Retrieving NOUNs from vocabulary, dataframe has got {len(reviews)} reviews.")
    reviews['nouns'] = reviews['text'].apply(lambda row: retrieve_noun_vocabulary(row, nlp))
    total_seconds = round(time.time() - start_time, 2)
    logger.info(f"NOUNs retrieved, took {str(timedelta(seconds=int(total_seconds)))} sec -> {total_seconds / len(reviews):.4f} sec/review")

    # Save the processed DataFrame to a CSV file
    logger.info("Writing to CSV file")
    reviews.to_csv(os.path.join(target_path, city, f + '.csv'))

    return reviews


def preprocess_text(text, nlp):
    # Step 1: Remove unwanted characters (punctuation, special characters, and numbers)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    # Step 2: Tokenization using NLTK's word tokenizer
    tokens = nltk.word_tokenize(text)

    # Step 3: Lowercase and filter out non-alphabetical tokens
    tokens = [token.lower() for token in tokens if token.isalpha()]

    # Step 4: Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Step 5: Lemmatization using SpaCy's NLP model
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc if token.lemma_ != "-PRON-"]  # Avoid pronouns

    # Step 6: Filter out short tokens
    tokens = [token for token in tokens if len(token) > 2]

    return tokens


def load_items_df(city, f):
    if os.path.exists(os.path.join(target_path, city, f + ".csv")):
        return pd.read_csv(os.path.join(target_path, city, f + ".csv"), encoding='utf-8')
    elif os.path.exists(os.path.join(source_path, city, f + ".pkl")):
        return load_pkl_and_save_as_csv(city, f)


def load_users_df(city, f):
    if os.path.exists(os.path.join(target_path, city, f + ".csv")):
        return pd.read_csv(os.path.join(target_path, city, f + ".csv"), encoding='utf-8')
    elif os.path.exists(os.path.join(source_path, city, f + ".pkl")):
        return load_pkl_and_save_as_csv(city, f)


def load_reviews_df(city, f, lang="en", from_date="2018-01-01", to_date="2023-01-01"):
    df = pd.DataFrame()
    if os.path.exists(os.path.join(target_path, city, f + ".csv")):
        df = pd.read_csv(os.path.join(target_path, city, f + ".csv"), encoding='utf-8')
    elif os.path.exists(os.path.join(source_path, city, f + ".pkl")):
        df = load_data_and_save_as_csv(city, f, lang, from_date=from_date, to_date=to_date)

    df = df[df['language'] == lang]
    df = df[(df['date'] >= from_date) & (df['date'] < to_date)]
    return df


def retrieve_noun_vocabulary(text, nlp):
    noun_tokens = set([])

    text = nlp(" ".join(text))
    for token in text:
        if token.pos_ == 'NOUN':
            noun_tokens.add(str(token).lower())
    return list(noun_tokens)
