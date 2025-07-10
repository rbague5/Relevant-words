import json
import os
import re
from collections import Counter
from statistics import mean

import torch

from config import top_n_restaurants
from preprocessing import load_reviews_df
from utils import logger

from sentence_transformers import SentenceTransformer, util
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


def get_cluster_words(city, path):
        clusters = {}

        for metric_folder in os.listdir(path):
            metric_path = os.path.join(path, metric_folder)
            if os.path.isdir(metric_path):
                clusters[metric_folder] = {}
                for filename in os.listdir(metric_path):
                    file_path = os.path.join(metric_path, filename)
                    if filename.endswith(".txt"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            clusters[metric_folder][filename] = f.read().strip().splitlines()

        return clusters


def calculate_label_confidence_cohesion(model, clusters_words_metrics, v=True):
    """
    Calculates confidence and cohesion scores for word clusters across multiple metrics.

    This function evaluates the semantic quality of clusters by computing:
    - **Confidence**: Cosine similarity between the cluster centroid and the word closest to it (the "best label").
    - **Cohesion**: Average cosine similarity between the centroid and all words in the cluster.

    Higher values for both confidence and cohesion indicate better, more semantically consistent clusters.

    Notes:
    -----
    - Confidence is based on the similarity of the best label (most central word) to the cluster centroid.
    - Cohesion reflects how semantically tight the cluster is around its centroid.
    - Both metrics use cosine similarity.
    """

    metrics_summary = {}

    for metric, cluster_words in clusters_words_metrics.items():
        logger.info(f"\n=== Metric: {metric} ===") if v else None

        confidences = []
        cohesions = []

        for cluster_id, words in cluster_words.items():
            logger.info(f"\nCluster {cluster_id}: {words}") if v else None

            if len(words) == 0:
                continue

            # Get embeddings
            embeddings = model.encode(words, convert_to_tensor=True)
            # Compute centroid
            centroid = embeddings.mean(dim=0)
            # Cosine similarities between centroid and each word
            sims = util.pytorch_cos_sim(centroid, embeddings)[0]  # shape: [num_words]
            # Best label (word closest to centroid)
            best_idx = torch.argmax(sims).item()
            best_label = words[best_idx]
            confidence = sims[best_idx].item()

            # Cohesion: average similarity of all words to centroid
            cohesion = sims.mean().item()

            logger.info(f"  → Best Label: '{best_label}'") if v else None
            logger.info(f"  → Confidence Score: {confidence:.2f}") if v else None
            logger.info(f"  → Cohesion Score:  {cohesion:.2f}") if v else None

            confidences.append(confidence)
            cohesions.append(cohesion)

        # Compute mean values
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0
        mean_cohesion = sum(cohesions) / len(cohesions) if cohesions else 0

        logger.info(f"\n>>> Mean Confidence for {metric}: {mean_confidence:.2f}") if v else None
        logger.info(f">>> Mean Cohesion for {metric}:  {mean_cohesion:.2f}") if v else None

        metrics_summary[metric] = {
            "mean_confidence": mean_confidence,
            "mean_cohesion": mean_cohesion
        }

    return metrics_summary

def get_best_keys_per_score(metric_summary):
    best_confidence_value = float('-inf')
    best_confidence_keys = []

    best_cohesion_value = float('-inf')
    best_cohesion_keys = []

    for key, scores in metric_summary.items():
        confidence = scores.get('mean_confidence', float('-inf'))
        cohesion = scores.get('mean_cohesion', float('-inf'))

        # Para mean_confidence
        if confidence > best_confidence_value:
            best_confidence_value = confidence
            best_confidence_keys = [key]
        elif confidence == best_confidence_value:
            best_confidence_keys.append(key)

        # Para mean_cohesion
        if cohesion > best_cohesion_value:
            best_cohesion_value = cohesion
            best_cohesion_keys = [key]
        elif cohesion == best_cohesion_value:
            best_cohesion_keys.append(key)

    return {
        'best_mean_confidence': (best_confidence_keys, best_confidence_value),
        'best_mean_cohesion': (best_cohesion_keys, best_cohesion_value),
    }



def main_cities_topics(data, city, restaurant_id=None, v=True):
    """
    Extract keywords from restaurant reviews and label predefined clusters.
    """
    logger.info(f"Extracting keyword phrases for city: {city.title()}")

    if restaurant_id is not None:
        cluster_words_path = os.path.join("results_by_restaurant", city, restaurant_id, "results", "topics")
    else:
        cluster_words_path = os.path.join("results_by_city", city, "results", "topics")

    clusters_words_by_metric = get_cluster_words(city, path=cluster_words_path)
    logger.info(f"Cluster words: {clusters_words_by_metric}") if v else None

    model = SentenceTransformer('all-MiniLM-L6-v2')

    metric_summary = calculate_label_confidence_cohesion(model, clusters_words_by_metric, v)
    logger.info(metric_summary) if v else None

    best_keys = get_best_keys_per_score(metric_summary)

    metric_summary['best_mean_confidence'] = best_keys['best_mean_confidence']
    metric_summary['best_mean_cohesion'] = best_keys['best_mean_cohesion']
    metric_summary['best_mean_metrics'] = mean([best_keys['best_mean_confidence'][1], best_keys['best_mean_cohesion'][1]])

    logger.info(f"Best mean_confidence: {metric_summary['best_mean_confidence']}") if v else None
    logger.info(f"Best mean_cohesion: {metric_summary['best_mean_cohesion']}") if v else None
    logger.info(f"Mean metric: {metric_summary['best_mean_metrics']}") if v else None

    if restaurant_id is not None:
        output_path = os.path.join("results_by_restaurant", city, restaurant_id, "metrics", "cluster_topic_metrics.json")
    else:
        output_path = os.path.join("results_by_city", city, "metrics", "cluster_topic_metrics.json")

    with open(output_path, "w") as f:
        json.dump(metric_summary, f, indent=4)

    return best_keys


def get_most_common_metrics(best_city_summary, city, v):
    all_metrics = []
    for metrics_list, val in best_city_summary.values():
        all_metrics.extend(metrics_list)

    metric_counter = Counter(all_metrics)

    if not metric_counter:
        logger.warning(f"No metrics found for {city}") if v else None
        return None

    max_count = max(metric_counter.values())
    most_common_metrics = [metric for metric, count in metric_counter.items() if count == max_count]

    logger.info(f"Most common metric(s) for {city} with count {max_count}: {most_common_metrics}") if v else None

    # Return a list of tied metrics (or a single metric if only one)
    return most_common_metrics if len(most_common_metrics) > 1 else most_common_metrics[0]


def count_metrics_across_cities(most_common_metrics_per_city, ignore_suffix, v):
    base_metric_counter = Counter()

    for city, metrics in most_common_metrics_per_city.items():
        # Normalize metrics to a list
        if isinstance(metrics, str):
            metrics = [metrics]

        for metric in metrics:
            # Remove _positive or _negative suffix
            if ignore_suffix:
                base_metric = re.sub(r'_(positive|negative)$', '', metric)
            else:
                base_metric = metric
            base_metric_counter[base_metric] += 1

    if ignore_suffix:
        logger.info(f"Metric counts ignoring suffix: {dict(base_metric_counter)}") if v else None
    else:
        logger.info(f"Metric counts: {dict(base_metric_counter)}") if v else None

    if base_metric_counter:
        # Sort metrics by frequency (most common first)
        sorted_metrics = base_metric_counter.most_common()

        # Print full ranking
        if v:
            if ignore_suffix:
                logger.info("Ranking of metrics (ignoring suffix):")
            else:
                logger.info("Ranking of metrics:")
            for rank, (metric, count) in enumerate(sorted_metrics, 1):
                logger.info(f"{rank}. {metric} - {count} times")

        most_common_metric, count = sorted_metrics[0]
        if ignore_suffix:
            logger.info(f"Most common metric overall (ignoring suffix): {most_common_metric} (appears {count} times)") if v else None
        else:
            logger.info(f"Most common metric overall: {most_common_metric} (appears {count} times)") if v else None
        return most_common_metric, count
    else:
        if v:
            logger.warning("No metrics found in overall counting.")
        return None, 0


def main_analysis_by_city(data, city):
    best_city_summary = main_cities_topics(data, city, v=verbose)
    logger.info(f"Best metrics of {city}: {best_city_summary}")
    most_common_metric = get_most_common_metrics(best_city_summary, city, v=verbose)
    most_common_metrics_per_city[city] = most_common_metric

    return most_common_metrics_per_city

def main_analysis_by_restaurant(data, city):
    most_commented_restaurants = data['itemId'].value_counts()
    most_common_metrics_per_city = {}
    for restaurant_id in most_commented_restaurants.head(top_n_restaurants).index:
        logger.info(f"Doing topic clustering for city: {city} and restaurant id: {restaurant_id}")

        best_city_summary = main_cities_topics(data, city, str(restaurant_id), v=verbose)
        logger.info(f"Best metrics of restaurant {restaurant_id} in {city}: {best_city_summary}")
        most_common_metric = get_most_common_metrics(best_city_summary, city, v=verbose)
        if city not in most_common_metrics_per_city:
            most_common_metrics_per_city[city] = {}

        most_common_metrics_per_city[city][restaurant_id] = most_common_metric

    return most_common_metrics_per_city

if __name__ == "__main__":
    # ["gijon", "moscow", "madrid", "istanbul", "barcelona"]
    most_common_metrics_per_city = {}
    most_common_metrics_per_city_all = {}
    verbose = True
    for city in ["moscow", "madrid", "istanbul", "barcelona"]:
        logger.info(f"Checking city: {city.title()}")

        logger.info(f"Creating reviews data")
        data = load_reviews_df(city, "reviews", lang="en", from_date="2018-01-01", to_date="2023-01-01")

        logger.info(f"Data of {city} loaded, in total there are {len(data.index)} reviews") if verbose else None
        logger.info(f"N. of users: {data['userId'].nunique()}") if verbose else None
        logger.info(f"N. of restaurants: {data['itemId'].nunique()}") if verbose else None
        logger.info(f"N. of reviews: {data['reviewId'].nunique()}") if verbose else None
        logger.info(f"N. of rating: {data['rating'].value_counts().sort_index(ascending=False)}") if verbose else None
        logger.info(f"Review dates between: {data['date'].min()} and {data['date'].max()}") if verbose else None

        # most_common_metrics_per_city = main_analysis_by_city(data, city)
        most_common_metrics_per_city_all = main_analysis_by_restaurant(data, city)

    logger.info("")
    _, _ = count_metrics_across_cities(most_common_metrics_per_city,ignore_suffix=True, v=True)
    logger.info("")
    _, _ = count_metrics_across_cities(most_common_metrics_per_city, ignore_suffix=False, v=True)
