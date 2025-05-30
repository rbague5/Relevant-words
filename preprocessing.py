import time
from datetime import timedelta
import pandas as pd

from config import target_path, source_path, load_spacy, load_nltk
from utils import logger
import os
import pickle
import re
from nltk.corpus import stopwords, words


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

    load_nltk()
    nlp = load_spacy()
    reviews = reviews.drop_duplicates(subset=['reviewId']).reset_index(drop=True)
    # Preprocess text in parallel
    start_time = time.time()
    logger.info("Preprocessing text")
    stop_words = set(stopwords.words('english'))
    english_words = set(words.words())

    reviews['text'] = reviews['text'].apply(lambda row: preprocess_text(row, nlp, stop_words=stop_words, english_words=english_words))
    total_seconds = round(time.time() - start_time, 2)
    logger.info(f"Text preprocessed, took {str(timedelta(seconds=int(total_seconds)))} sec -> {total_seconds / len(reviews):.4f} sec/review")

    # Retrieve nouns in parallel
    start_time = time.time()
    logger.info(f"Retrieving NOUNs from vocabulary, dataframe has got {len(reviews)} reviews.")
    reviews['nouns'] = reviews['text'].apply(lambda row: retrieve_noun_vocabulary(row, nlp, stop_words=stop_words, english_words=english_words))

    total_seconds = round(time.time() - start_time, 2)
    logger.info(f"NOUNs retrieved, took {str(timedelta(seconds=int(total_seconds)))} sec -> {total_seconds / len(reviews):.4f} sec/review")

    # Save the processed DataFrame to a CSV file
    logger.info("Writing to CSV file")
    reviews.to_csv(os.path.join(target_path, city, f + '.csv'))

    return reviews


def preprocess_text(text, nlp, stop_words, english_words):
    # Step 1: Basic cleaning
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII

    # Step 2: Process with spaCy for lemmatization and POS tagging
    doc = nlp(text)

    # Step 3: Extract lemmas, lowercase, filter pronouns
    tokens = [token.lemma_.lower() for token in doc if token.lemma_ != "-PRON-" and token.is_alpha]

    # Step 4: Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Step 5: Remove non-English words
    tokens = [token for token in tokens if token in english_words]

    # Step 6: Filter out short tokens
    tokens = [token for token in tokens if len(token) > 2]

    return tokens


def retrieve_noun_vocabulary(tokens, nlp, stop_words=None, english_words=None):
    doc = nlp(" ".join(tokens))  # Create a new doc from preprocessed tokens

    # Extract lemmatized nouns
    nouns = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ == 'NOUN'
           and token.lemma_ != "-PRON-"
           and token.is_alpha
    ]
    # Filter out short nouns
    nouns = [noun for noun in nouns if len(noun) > 2]

    # Remove stop words if provided
    if stop_words:
        nouns = [noun for noun in nouns if noun not in stop_words]

    # Keep only English words if provided
    if english_words:
        nouns = [noun for noun in nouns if noun in english_words]

    return nouns

def load_items_df(city, f):
    if os.path.exists(os.path.join(target_path, city, f + ".csv")):
        return pd.read_csv(os.path.join(target_path, city, f + ".csv"), encoding='utf-8')
    elif os.path.exists(os.path.join(source_path, city, f + ".pkl")):
        return load_pkl_and_save_as_csv(city, f)
    else:
        raise Exception(f"Path of items for city {city} doesn't exist.")


def load_users_df(city, f):
    if os.path.exists(os.path.join(target_path, city, f + ".csv")):
        return pd.read_csv(os.path.join(target_path, city, f + ".csv"), encoding='utf-8')
    elif os.path.exists(os.path.join(source_path, city, f + ".pkl")):
        return load_pkl_and_save_as_csv(city, f)
    else:
        raise Exception(f"Path of users for city {city} doesn't exist.")


def load_reviews_df(city, filename, lang="en", from_date="2018-01-01", to_date="2023-01-01"):
    csv_path = os.path.join(target_path, city, f"{filename}.csv")
    pkl_path = os.path.join(source_path, city, f"{filename}.pkl")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding='utf-8')
    elif os.path.exists(pkl_path):
        df = load_data_and_save_as_csv(city, filename, lang, from_date=from_date, to_date=to_date)
    else:
        return pd.DataFrame()  # Optional: fallback if neither file exists

    # Apply filters efficiently
    df = df.loc[(df['language'] == lang) & (df['date'] >= from_date) & (df['date'] < to_date)]

    return df
