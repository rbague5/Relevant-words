import os
import json
import shutil
from pathlib import Path

from utils import logger

def extract_best_results(city_path, class_type):
    best_results_path = os.path.join(city_path, f"best_overall_{class_type}_metrics.json")

    with open(best_results_path, "r") as f:
        data = json.load(f)
    logger.info(f"Best Overall Metrics for {Path(city_path).name.title()}: {data}")

    best_results = {}
    for metric_name, value in data.items():
        logger.info(f"Checking metric {metric_name}")
        value = eval(value)
        if isinstance(value, tuple):
            best_config_path = value[0]
            best_number_clusters = value[1]
            best_metric_value = value[2]
            logger.info(f"Best config path: {best_config_path}")
            best_config_figures_path = os.path.join(city_path, best_config_path, "results", "figures", f"{metric_name}_{class_type}_{best_number_clusters}.png")
            best_config_topics_path = os.path.join(city_path, best_config_path, "results", "topics", f"{metric_name}_{class_type}")

            best_results[metric_name] = {
                "best_config_figures_path": best_config_figures_path,
                "best_config_topics_path": best_config_topics_path
            }
        else:
            raise Exception("Value is not a tuple")

    return best_results


if __name__ == "__main__":
    base_path = Path("C:/Users/Ants/Documents/Doctorat/Article 2/Relevant-words/results_by_city")
    filtered_results_path = Path("C:/Users/Ants/Documents/Doctorat/Article 2/Relevant-words/results_by_city_paper")
    all_results = {}
    for city in ["moscow", "madrid", "istanbul", "barcelona"]:
        logger.info(f"Checking city: {city.title()}")
        all_results[city] = {}
        for class_type in ["positive", "negative"]:
            logger.info(f"Class type: {class_type}")
            city_path = os.path.join(base_path, city)
            different_combinations_names = os.listdir(city_path)
            logger.info(f"Different combinations: {different_combinations_names}")
            all_results[city][class_type] = {}
            for different_config in different_combinations_names:
                if ".json" in different_config:
                    continue
                all_results[city][class_type][different_config] = {}
                all_metrics_path = os.path.join(city_path, different_config, "metrics", class_type,"all_metrics_metrics.json")
                with open(all_metrics_path, "r") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if "scci" in v and isinstance(v["scci"], dict):
                            v["scci"] = v["scci"]["scci"]
                all_results[city][class_type][different_config] = data
    with open("all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    best_results = {}
    maximize_metrics = {"semantic_coherence", "scci", "inter_distance", "silhouette_score", "npmi"}
    minimize_metrics = {"intra_distance"}
    for city, sentiment_data in all_results.items():
        best_results[city] = {}
        for class_type, configs in sentiment_data.items():
            best_results[city][class_type] = {}
            metric_best = {}
            for config_name, k_values in configs.items():
                for k, metrics in k_values.items():
                    for metric_name, metric_value in metrics.items():

                        # Initialize metric record if not seen before
                        if metric_name not in metric_best:
                            metric_best[metric_name] = {
                                "config": config_name,
                                "k": k,
                                "value": metric_value
                            }
                            continue

                        # Decide if new value is better
                        if metric_name in maximize_metrics:
                            if metric_value > metric_best[metric_name]["value"]:
                                metric_best[metric_name] = {
                                    "config": config_name,
                                    "k": k,
                                    "value": metric_value
                                }
                        elif metric_name in minimize_metrics:
                            if metric_value < metric_best[metric_name]["value"]:
                                metric_best[metric_name] = {
                                    "config": config_name,
                                    "k": k,
                                    "value": metric_value
                                }
            # Save for this class type
            best_results[city][class_type] = metric_best

    with open("best_results.json", "w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=4)

    for city, sentiment_data in best_results.items():
        for class_type in ["positive", "negative"]:
            for metric, values in best_results[city][class_type].items():
                figure_src = Path(os.path.join(base_path, city, values["config"], "results", "figures", f"{metric}_{class_type}_{values['k']}.png"))
                topics_src = Path(os.path.join(base_path, city, values["config"], "results", "topics", f"{metric}_{class_type}"))

                dest_img = Path(os.path.join(filtered_results_path, city, "results", "figures"))
                dest_topics = Path(os.path.join(filtered_results_path, city, "results", "topics"))
                os.makedirs(dest_img, exist_ok=True)

                # Copy figure file
                if figure_src.exists():
                    shutil.copy(figure_src, dest_img)
                else:
                    print(f"⚠️ Figure not found: {figure_src}")

                # Copy topics directory
                dest_topics = os.path.join(dest_topics, f"{metric}_{class_type}")
                os.makedirs(dest_topics, exist_ok=True)
                if topics_src.exists():
                    for item in topics_src.iterdir():
                        if item.is_file():
                            shutil.copy(item, dest_topics)
                        elif item.is_dir():
                            shutil.copytree(item, dest_topics / item.name, dirs_exist_ok=True)
                else:
                    print(f"⚠️ Topics folder missing: {topics_src}")
