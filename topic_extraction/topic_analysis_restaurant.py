import json
import logging
import math
import os
import re

from config import top_n_nearest_points
from generate_paper_figures import generate_plots
from utils import logger


def load_jsonl1(path: str, key: str = "responses"):
    topics = []
    pattern = re.compile(r"\[\d+\]\s*(.*?):\s*(.*?)(?=\n*\[\d+\]|$)", re.DOTALL)
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                topics_info = json.loads(line)
                if topics_info[key] == "None":
                    topics_info[key] = None
                else:
                    matches = pattern.findall(topics_info[key])

                    if matches:
                        parsed_topics = []
                        for match in matches:
                            topic_name = match[0].strip()
                            description = match[1].strip() if match[1] else None

                            parsed_topics.append({
                                "topic_id": json.loads(line)["id"],
                                "topic": topic_name,
                                "description": description
                            })

                        topics_info[key] = parsed_topics
                    else:
                        topics_info[key] = None
                topics.append(topics_info)
    return topics


def load_jsonl2(path: str, key: str = "responses"):
    topics = []
    pattern = re.compile(
        r"^\s*\[(\d+)\]\s*([^\(:\n]+)"  # capture level [1] or [2] and topic name
        r"(?:\s*\(Document:\s*([^\)]+)\))?"  # optional (Document: x, y)
        r"(?:\s*:\s*(.*?))?"  # optional description
        r"(?=\n\s*\[\d+\]|\Z)",  # stop before next [number] or end of text
        re.DOTALL | re.MULTILINE
    )
    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            if line.strip():
                topics_info = json.loads(line)
                if topics_info[key] == "None":
                    topics_info[key] = None
                else:
                    matches = pattern.findall(topics_info[key])

                    if matches:
                        parsed_topics = []
                        parent_topic = None
                        for match in matches:
                            level = match[0].strip()
                            topic_name = match[1].strip()
                            document = match[2].strip() if match[2] else None
                            description = match[3].strip() if match[3] else None
                            parent_topic = parent_topic if level != '1' else None
                            if len(topics_info["text"].split("\n")) == 3:
                                txt = topics_info["text"].split("\n")[1].strip()
                            else:
                                if document == "1":
                                    txt = topics_info["text"].split("\n")[1].strip()
                                elif document == "2":
                                    txt = topics_info["text"].split("\n")[3].strip()
                                else:
                                    txt = None

                            parsed_topics.append({
                                "level": int(level),
                                "parent_topic": parent_topic,
                                "topic": topic_name,
                                "document": document,
                                "description": description,
                                "text": txt
                            })
                            if level == '1':
                                parent_topic = topic_name

                        topics_info[key] = parsed_topics
                    else:
                        topics_info[key] = None
                topics.append(topics_info)
    return topics


def generate_imgs_latex(city: str, restaurant_id: str, metrics: list):
    metrics[0] = metrics[0].replace("_", "-")
    metrics[1] = metrics[1].replace("_", "-")
    latex_code = ["", ""]
    latex_code.append(r"\begin{figure}[htbp]")
    latex_code.append(r"    \centering")
    latex_code.append(r"    \begin{subfigure}[b]{0.49\textwidth}")
    latex_code.append(r"        \centering")
    latex_code.append(rf"        \includegraphics[width=\textwidth]{{images/figures_per_restaurant/{city.lower()}/{restaurant_id}/{metrics[0].lower().replace('-', '_')}_negative.png}}")
    latex_code.append(rf"        \caption{{{metrics[0].title()} clusters $\left(C_{{R_{{{restaurant_id}}}}}^{{-}}\right)$ {city.title()}}}")
    latex_code.append(r"    \end{subfigure}")
    latex_code.append(r"    \begin{subfigure}[b]{0.49\textwidth}")
    latex_code.append(r"        \centering")
    latex_code.append(rf"        \includegraphics[width=\textwidth]{{images/figures_per_restaurant/{city.lower()}/{restaurant_id}/{metrics[0].lower().replace('-', '_')}_positive.png}}")
    latex_code.append(rf"        \caption{{{metrics[0].title()} clusters $\left(C_{{R_{{{restaurant_id}}}}}^{{+}}\right)$ {city.title()}}}")
    latex_code.append(r"    \end{subfigure}")
    latex_code.append(rf"    \caption{{Top clusters in {city.title()} based on {metrics[0].title()}.}}")
    latex_code.append(rf"    \label{{fig:metrics_{city.lower()}_{metrics[0].lower().replace('-', '_')}}}")
    latex_code.append(r"\end{figure}")
    latex_code.append(r"\begin{figure}[htbp]")
    latex_code.append(r"    \centering")
    latex_code.append(r"    \begin{subfigure}[b]{0.49\textwidth}")
    latex_code.append(r"        \centering")
    latex_code.append(rf"        \includegraphics[width=\textwidth]{{images/figures_per_restaurant/{city.lower()}/{restaurant_id}/{metrics[1].lower().replace('-', '_')}_negative.png}}")
    latex_code.append(rf"        \caption{{{metrics[1].title()} clusters $\left(C_{{R_{{{restaurant_id}}}}}^{{-}}\right)$ {city.title()}}}")
    latex_code.append(r"    \end{subfigure}")
    latex_code.append(r"    \begin{subfigure}[b]{0.49\textwidth}")
    latex_code.append(r"        \centering")
    latex_code.append(rf"        \includegraphics[width=\textwidth]{{images/figures_per_restaurant/{city.lower()}/{restaurant_id}/{metrics[1].lower().replace('-', '_')}_positive.png}}")
    latex_code.append(rf"        \caption{{{metrics[1].title()} clusters $\left(C_{{R_{{{restaurant_id}}}}}^{{+}}\right)$ {city.title()}}}")
    latex_code.append(r"    \end{subfigure}")
    latex_code.append(rf"    \caption{{Top clusters in {city.title()} based on {metrics[1].title()}.}}")
    latex_code.append(rf"    \label{{fig:metrics_{city.lower()}_{metrics[1].lower().replace('-', '_')}}}")
    latex_code.append(r"\end{figure}")

    return latex_code

data = []

# ["moscow", "madrid", "istanbul", "barcelona"]
for restaurant_id in ["691623", "2419130", "7122712"]:
    for city in ["moscow"]:
# for restaurant_id in ["1027419", "6367286", "17340009"]:
#     for city in ["madrid"]:
# for restaurant_id in ["1010030", "1644684", "1749881"]:
#     for city in ["istanbul"]:
# for restaurant_id in ["1082035", "3179219", "11906544"]:
#     for city in ["barcelona"]:
        summary = {}
        for metric in ["inter_distance_negative", "inter_distance_positive", "intra_distance_negative", "intra_distance_positive", "scci_negative", "scci_positive", "semantic_coherence_negative", "semantic_coherence_positive", "silhouette_score_negative", "silhouette_score_positive"]:
            input_path_generation_1 = os.path.join("data_restaurants", "output", f"top_{top_n_nearest_points}", city, restaurant_id, metric, "generation_1.jsonl")
            input_path_generation_2 = os.path.join("data_restaurants", "output", f"top_{top_n_nearest_points}", city, restaurant_id, metric, "generation_2.jsonl")
            topics_generation_1 = load_jsonl1(input_path_generation_1)
            topics_generation_2 = load_jsonl2(input_path_generation_2, key="topics")

            final_result = []
            subtopics = []
            for topic_1 in topics_generation_1:
                if topic_1["responses"] == None:
                    final_result.append(topic_1)
                    continue
                else:
                    filtered_topics = []

                    for t in topics_generation_2:
                        # Extract the second line safely.
                        line_parts = t["text"].split("\n")
                        line_parts = [lp for lp in line_parts if lp.strip() != ""]
                        if len(line_parts) > 1 and line_parts[1] == topic_1["text"] and len(line_parts) == 2:
                            filtered_topics.append(t)
                        else:
                            filtered_topics.append(t)

                    for topic_2 in filtered_topics:
                        topics = topic_2["topics"]
                        doc_subtopics = {}
                        current_doc = ""
                        text = topic_2["text"].split("\n")
                        for doc in text:
                            if "" == doc:
                                continue
                            if "Document " in doc:
                                doc_subtopics[doc] = ""
                                current_doc = doc
                            else:
                                if topic_1["text"] == doc:
                                    doc_subtopics[current_doc] = doc
                        if len(doc_subtopics) > 0:
                            subtopics.extend([t["topic"] for t in topic_2["topics"] if t["level"] != 1])
                        if len(topic_2["topics"]) == 1:
                            subtopics.extend([topic_2["topics"][0]["topic"]])

                        for subtopic in topics:
                            if subtopic["level"] != 1 and subtopic["text"] == topic_1["text"]:
                                value = f"{subtopic['topic']}: {subtopic['description']}"
                                if "subtopics" not in topic_1:
                                    topic_1["subtopics"] = []
                                topic_1["subtopics"].append(value)

                    final_result.append(topic_1)
            # c = Counter(topics)
            summary[metric] = {
                "#_clusters": len(final_result),
                "#_non_assigned": sum(1 for topic in final_result if topic["responses"] is None),
                "#_different_topics": len(list(set(subtopics))),
                "topics": list(set(subtopics)),
                "topics_details": final_result,
                # "most_common_topics": Counter(
                #     t.get("subtopics", t.get("responses", "")) for t in final_result
                # )
            }
        pretty_json = json.dumps(summary, indent=4)

        # logger.info(pretty_json)
        metrics_mapping = {
            "inter_distance_negative": "Inter-distance $C^{-}$",
            "inter_distance_positive": "Inter-distance $C^{+}$",
            "intra_distance_negative": "Intra-distance $C^{-}$",
            "intra_distance_positive": "Intra-distance $C^{+}$",
            "scci_negative": "SCCI $C^{-}$",
            "scci_positive": "SCCI $C^{+}$",
            "semantic_coherence_negative": "Semantic Coherence $C^{-}$",
            "semantic_coherence_positive": "Semantic Coherence $C^{+}$",
            "silhouette_score_negative": "Silhouette Score $C^{-}$",
            "silhouette_score_positive": "Silhouette Score $C^{+}$"
        }

        latex = []
        latex.append(r"\begin{table}[h!]")
        latex.append(r"\centering")
        latex.append(r"\begin{tabular}{|l|c|c|c|c|c|}")
        latex.append(r"\hline")
        latex.append(r"Metric & \# Clusters & \# Non assigned topics & \# Diff. Topics & CQS & Mean \\")
        latex.append(r"\hline")

        total_clusters = 0
        total_non_assigned = 0
        total_diff_topics = 0
        h = 0
        cqs_values = []
        metric_cluster_results = {}
        means_sqs_metrics = {}
        for metric, values in summary.items():
            topics_str = ", ".join(values["topics"])

            # Extract values
            clusters = values["#_clusters"]
            non_assigned = values["#_non_assigned"]
            diff_topics = values["#_different_topics"]
            metric_cluster_results[metric] = values["topics_details"]

            # Compute CQS score
            if clusters > 0:
                AQ = 1 - (non_assigned / clusters)
                DQ = diff_topics / clusters
                if (AQ + DQ) > 0:
                    cqs = (2 * AQ * DQ / (AQ + DQ)) * math.log(clusters + 1)
                else:
                    cqs = 0
            else:
                cqs = 0

            # Add row to LaTeX with new CCQS column
            latex.append(
                f"{metrics_mapping[metric]} & "
                f"{clusters} & "
                f"{non_assigned} "
                f"({round((non_assigned / clusters) * 100, 2)}\\%) & "
                f"{diff_topics} & "
                f"{round(cqs, 2)} & \\\\"
            )
            cqs_values.append(cqs)

            # accumulate totals
            total_clusters += clusters
            total_non_assigned += non_assigned
            total_diff_topics += values["#_different_topics"]

            h += 1
            if h == 2:
                mean_cqs_metric = round((sum(cqs_values) / len(cqs_values)), 2)
                latex[-2] = latex[-2].replace(r"\\", "") + r"\multirow{2}{*}[-2pt]{" + str(mean_cqs_metric) + r"} \\"
                latex.append(r"\hline")
                # latex.append(rf"Mean &  &  &  & {round((sum(cqs_values) / len(cqs_values)), 2)} \\")
                # latex.append(r"\hline")
                # latex.append(r"\hline")
                means_sqs_metrics[metric.replace("_negative", "").replace("_positive", "")] = round(
                    (sum(cqs_values) / len(cqs_values)), 2)
                h = 0
                cqs_values = []

        percent_non_asigned = round((total_non_assigned / total_clusters) * 100, 2)
        latex.append(
            f"Total & "
            f"{total_clusters} & "
            f"{total_non_assigned} "
            f"({format(percent_non_asigned)}\\%) & "
            f"{total_diff_topics} & N/A & N/A \\\\"
        )
        latex.append(r"\hline")
        latex.append(r"\end{tabular}")
        latex.append(r"\caption{Summary of Clustering Metrics of " + city.title() + " " + restaurant_id + "}")
        latex.append(r"\label{tab:summary_metrics_topics_" + city + "_" + restaurant_id + "}")
        latex.append(r"\end{table}")
        logger.info("\n".join(latex))
        top2_keys = [k for k, v in sorted(means_sqs_metrics.items(), key=lambda x: x[1], reverse=True)[:2]]
        # ---- Save original formatter ----
        original_formatter = logger.handlers[0].formatter

        # ---- Temporary formatter (only message, no date, no level) ----
        clean_formatter = logging.Formatter("%(message)s")
        logger.handlers[0].setFormatter(clean_formatter)

        # logger.info(f"The results of the topics of the clusters for the best metrics of the restaurant {restaurant_id} in {city.title()} are the following:\n")
        # for metric, value in metric_cluster_results.items():
        #     if metric.replace("_negative", "").replace("_positive", "") not in top2_keys:
        #         continue
        #     input_path_full_last_results = os.path.join("data_restaurants", f"results_top_{top_n_nearest_points}", city, restaurant_id, f"{metric}.json")
        #     with open(input_path_full_last_results, "r", encoding="utf-8") as f:
        #         full_last_results = json.load(f)
        #     logger.info(f"\n\\textbf{{{metric.replace('_',' ').title()}}}\n")
        #
        #     logger.info("\\begin{itemize}")
        #     no_topics_clusters = []
        #     for cluster in sorted(value, key=lambda x: x['id']):
        #         cluster.pop('text', None)
        #         cluster_id = cluster.get('id')
        #         for cluster_last_result in full_last_results:
        #             if cluster_last_result.get('id') == cluster_id:
        #                 explanation = cluster_last_result.get('explanation')
        #                 break
        #         responses = cluster.get('responses')
        #         subtopics = list(set(list(cluster.get('subtopics', []))))
        #         if responses:
        #             topics = ", ".join([r['topic'] + ": " + r['description'] for r in responses if 'topic' in r])
        #             if subtopics:
        #                 logger.info(f"    \\item \\textbf{{Cluster {cluster_id}}}: {topics}")
        #                 logger.info("    \\begin{itemize}")
        #                 logger.info(f"        \\item \\textbf{{Subtopics:}}")
        #                 for subt in subtopics:
        #                     logger.info(f"        \\item {subt}")
        #                 logger.info(f"        \\item \\textbf{{Explanation:}} {explanation.strip()}")
        #                 logger.info("    \\end{itemize}")
        #             else:
        #                 logger.info(f"    \\item \\textbf{{Cluster {cluster_id}}}: {topics}")
        #         else:
        #             no_topics_clusters.append(str(cluster_id))
        #     if len(no_topics_clusters) == 1:
        #         logger.info(f"    \\item \\textbf{{Cluster {', '.join(no_topics_clusters)}}}: Have no relevant topics.")
        #     else:
        #         logger.info(
        #             f"    \\item \\textbf{{Clusters {', '.join(no_topics_clusters)}}}: Have no relevant topics.")
        #     logger.info("\\end{itemize}")
        logger.info("\n".join(generate_imgs_latex(city, restaurant_id, top2_keys)))
        generate_plots(city, top2_keys[0], topics=summary, restaurant_id=restaurant_id, algorithm="umap")
        generate_plots(city, top2_keys[1], topics=summary, restaurant_id=restaurant_id, algorithm="umap")
        logger.handlers[0].setFormatter(original_formatter)

        # logger.info(f"\n--- SUMMARY STATISTICS {top_n_nearest_points} words ---")
        # logger.info(f"Total clusters: {total_clusters}")
        # logger.info(f"Total non-assigned: {total_non_assigned}")
        # logger.info(f"Total different topics entries: {total_diff_topics}")
        #
        # # percentage of non-assigned = total_non_assigned / total_clusters
        # percentage_non_assigned = (total_non_assigned / total_clusters) * 100
        # logger.info(f"Percentage non-assigned: {percentage_non_assigned:.2f}%")
