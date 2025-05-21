#!/usr/bin/env python3
from preprocessing import load_reviews_df
from utils import logger

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import yake
import numpy as np


def extract_phrases(texts, top_n=50):
    """
    Extract keyword phrases using YAKE from a list of review texts.
    """
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=top_n)
    all_phrases = set()
    for text in texts:
        try:
            keywords = kw_extractor.extract_keywords(text)
            for kw, _ in keywords:
                all_phrases.add(kw)
        except Exception:
            continue
    return list(all_phrases)


def label_clusters(clusters, candidate_phrases, model):
    """
    Assign the most relevant label (phrase) to each cluster based on semantic similarity.
    """
    cluster_embs = [model.encode(c) for c in clusters]
    cluster_centers = [np.mean(emb, axis=0) for emb in cluster_embs]
    phrase_embs = model.encode(candidate_phrases)

    for i, center in enumerate(cluster_centers):
        sims = cosine_similarity([center], phrase_embs)[0]
        best_idx = np.argmax(sims)
        cohesion = 1 - pairwise_distances(cluster_embs[i], metric="cosine").mean()

        print(f"\nCluster {i}:")
        print(f"  Words: {clusters[i]}")
        print(f"  → Best Label: '{candidate_phrases[best_idx]}'")
        print(f"  → Confidence Score: {sims[best_idx]:.2f}")
        print(f"  → Cohesion Score:  {cohesion:.2f}")


def get_cluster_words(city):
    pass



def main_cities_topics(data, city):
    """
    Extract keywords from restaurant reviews and label predefined clusters.
    """
    logger.info(f"Extracting keyword phrases for city: {city.title()}")

    texts = data["text"].dropna().tolist()
    candidate_phrases = extract_phrases(texts, top_n=30)
    logger.info(f"Extracted {len(candidate_phrases)} candidate phrases from reviews.")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    clusters_words = get_cluster_words(city)

    label_clusters(clusters_words, candidate_phrases, model)


if __name__ == "__main__":
    # ["gijon", "moscow", "madrid", "istanbul", "barcelona"]
    for city in ["moscow"]:
        logger.info(f"Checking city: {city.title()}")

        logger.info(f"Creating reviews data")
        data = load_reviews_df(city, "reviews", lang="en", from_date="2018-01-01", to_date="2023-01-01")

        logger.info(f"Data of {city} loaded, in total there are {len(data.index)} reviews")
        logger.info(f"N. of users: {data['userId'].nunique()}")
        logger.info(f"N. of restaurants: {data['itemId'].nunique()}")
        logger.info(f"N. of reviews: {data['reviewId'].nunique()}")
        logger.info(f"N. of rating: {data['rating'].value_counts().sort_index(ascending=False)}")
        logger.info(f"Review dates between: {data['date'].min()} and {data['date'].max()}")

        main_cities_topics(data, city)
        main_restaurant_cities_topics(data, city)
