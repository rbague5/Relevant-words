import os
import json
import re

from collections import Counter

from dotenv import load_dotenv
from topicgpt_python import *
import sys
from topicgpt_python import generate_topic_lvl1, assign_topics, correct_topics, generate_topic_lvl2

from config import top_n_nearest_points
from utils import logger

load_dotenv()


api = "openai"
model = "gpt-4o"
is_verbose = True
is_refine_topics = False
is_generate_subtopics = True

# Your existing code to load clusters
# for city in ["moscow", "madrid", "istanbul", "barcelona"]
# for restaurant_id in ["691623", "2419130", "7122712"]:
#     for city in ["moscow"]:
# for restaurant_id in ["1027419", "6367286", "17340009"]:
#     for city in ["madrid"]:
# for restaurant_id in ["1010030", "1644684", "1749881"]:
#     for city in ["istanbul"]:
for restaurant_id in ["1082035", "3179219", "11906544"]:
    for city in ["barcelona"]:
        summary = {}
        for metric in ["inter_distance_negative", "inter_distance_positive", "intra_distance_negative", "intra_distance_positive", "scci_negative", "scci_positive", "semantic_coherence_negative", "semantic_coherence_positive", "silhouette_score_negative", "silhouette_score_positive"]:
            base_path = os.path.join("..", f"results_by_restaurant_paper_top{top_n_nearest_points}", city, restaurant_id,"results", "topics", metric)

            cluster_files = sorted([f for f in os.listdir(base_path) if f.endswith(".txt")])
            clusters = {}
            for fname in cluster_files:
                topic_id = os.path.splitext(fname)[0]
                with open(os.path.join(base_path, fname), "r", encoding="utf-8") as f:
                    words = [w.strip() for w in f if w.strip()]
                    clusters[topic_id] = words

            logger.info(f"Loaded {len(clusters)} clusters.")

            # Convert your clusters to the required JSONL format
            output_dir = "data_restaurants/input"
            os.makedirs(output_dir, exist_ok=True)
            file = f"{city}_{restaurant_id}_{metric}_clusters.jsonl"
            data_sample = os.path.join(output_dir, file)

            with open(data_sample, "w", encoding="utf-8") as f:
                for topic_id, words in clusters.items():
                    # Join words into a document text
                    text = " ".join(words)
                    entry = {
                        "id": topic_id,
                        "text": text
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            logger.info(f"Created JSONL file: {data_sample}")

            os.makedirs(f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}", exist_ok=True)
            generation_prompt = "data_restaurants/prompt/generation_1.txt"
            generation_seed = "data_restaurants/input/seed_topics.txt"
            generation_output = f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}/generation_1.jsonl"
            generation_topic_output = f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}/generation_1.md"

            generate_topic_lvl1(
                api,
                model,
                data_sample,
                generation_prompt,
                generation_seed,
                generation_output,
                generation_topic_output,
                verbose=is_verbose,
            )

            refinement_prompt = "data_restaurants/prompt/refinement.txt"
            refinement_output = f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}/refinement.jsonl"
            refinement_topic_output = f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}/refinement.md"
            refinement_mapping_file = f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}/refinement_mapping.json"
            refinement_remove = False

            # Optional: Refine topics if needed
            if is_refine_topics:
                refine_topics(
                    api,
                    model,
                    refinement_prompt,
                    generation_output,
                    generation_topic_output,
                    refinement_topic_output,
                    refinement_output,
                    verbose=is_verbose,
                    remove=refinement_remove,
                    mapping_file=refinement_mapping_file
                )


            generation_2_prompt = "data_restaurants/prompt/generation_2.txt"
            generation_2_output = f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}/generation_2.jsonl"
            generation_2_topic_output = f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}/generation_2.md"

            # Optional: Generate subtopics
            if is_generate_subtopics:
                generate_topic_lvl2(
                    api,
                    model,
                    generation_topic_output,
                    generation_output,
                    generation_2_prompt,
                    generation_2_output,
                    generation_2_topic_output,
                    verbose=is_verbose,
                )

            assignment_prompt = "data_restaurants/prompt/assignment.txt"
            assignment_output = f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}/assignment.jsonl"

            if is_generate_subtopics:
                generation_topic_output = generation_2_topic_output

            # Assignment
            assign_topics(
                api,
                "gpt-4o-mini",
                data_sample,
                assignment_prompt,
                assignment_output,
                generation_topic_output,  # generation_topic_output, generation_2_topic_output TODO: change to generation_2 if you have subtopics, or config['refinement']['topic_output'] if you refined topics
                verbose=is_verbose,
            )

            correction_prompt = "data_restaurants/prompt/correction.txt"
            correction_output = f"data_restaurants/output/top_{top_n_nearest_points}/{city}/{restaurant_id}/{metric}/assignment_corrected.jsonl"
            # Correction
            correct_topics(
                api,
                "gpt-4o-mini",
                assignment_output,
                correction_prompt,
                generation_topic_output,  # generation_topic_output, generation_2_topic_output TODO: change to generation_2 if you have subtopics, or config['refinement']['topic_output'] if you refined topics
                correction_output,
                verbose=is_verbose,
            )

            data = []
            pattern = re.compile(r"\[\d+\]\s*(.*?):\s*(.*?)(?=\n*\[\d+\]|$)", re.DOTALL)

            topics = []
            # Convert .jsonl to .json (list)
            with open(correction_output, "r", encoding="utf-8") as infile:
                for line in infile:
                    if line.strip():
                        topics_info = json.loads(line)
                        responses = topics_info.get("responses", "").strip()

                        matches = pattern.findall(responses)

                        if matches:
                            for match in matches:
                                topic_name = match[0].strip()
                                explanation = match[1].strip()
                                extracted_data = {
                                    "id": topics_info.get("id"),
                                    "text": topics_info.get("text"),
                                    "prompted_docs": topics_info.get("prompted_docs"),
                                    "responses": topics_info.get("responses"),
                                    "topic": topic_name,
                                    "explanation": explanation
                                }
                                data.append(extracted_data)
                                topics.append(topic_name)
                        else:
                            # if no structured match, store raw response
                            data.append({
                                "id": topics_info.get("id"),
                                "text": topics_info.get("text"),
                                "prompted_docs": topics_info.get("prompted_docs"),
                                "responses": topics_info.get("responses"),
                                "topic": None,
                                "explanation": responses or None
                            })
                            topics.append("None")

            c = Counter(topics)
            summary[metric] = dict(c)
            # Generate output path safely
            output_path = os.path.join("data_restaurants", f"results_top_{top_n_nearest_points}", city, restaurant_id, f"{metric}.json")
            # Ensure output folder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Write list to JSON
            with open(output_path, "w", encoding="utf-8") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=2)

            logger.info(f"Converted {correction_output} to {output_path}")
        with open(os.path.join("data_restaurants", f"results_top_{top_n_nearest_points}", f"summary_{city}_{restaurant_id}.json"), "w", encoding="utf-8") as outfile:
            json.dump(summary, outfile, ensure_ascii=False, indent=2)
