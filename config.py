import os


def load_spacy():
    import spacy
    # spaCy config
    spacy_model = "en_core_web_lg"
    spacy.cli.download(spacy_model)
    nlp = spacy.load(spacy_model)
    return nlp


def load_nltk():
    import nltk
    # NLTK config
    nltk.download('stopwords')
    nltk.download('punkt')


columns_to_delete = ["images", "index", "language", "title", "url", "date"]

base_path = "./dataset"
source_path = os.path.join(base_path, "raw")
target_path = os.path.join(base_path, "input")

cities = ["gijon", "moscow", "madrid", "istanbul", "barcelona"]
rating_threshold = 40
top_n_restaurants = 5

top_n_nearest_points = 10

w2v_models_path_by_city = f"./models/word2vec/"
gmm_models_path_by_city = f"./models/gmm/"
topics_clusters_path_by_city = f"./results/topics/"
figures_path_by_city = f"./results/figures/"

w2v_models_path_by_restaurant = f"./models/word2vec/"
gmm_models_path_by_restaurant = f"./models/gmm/"
topics_clusters_path_by_restaurant = f"./results/topics/"
figures_path_by_restaurant = f"./results/figures/"

w2v_models_path_tf_itf_city = f"./models/word2vec/"
gmm_models_path_tf_itf_city = f"./models/gmm/"
topics_clusters_path_tf_itf_city = f"./results/topics/"
figures_path_tf_itf_city = f"./results/figures/"


positive_model_path_by_city = os.path.join(w2v_models_path_by_city, "positive")
negative_model_path_by_city = os.path.join(w2v_models_path_by_city, "negative")

positive_model_path_by_restaurant = os.path.join(w2v_models_path_by_restaurant, "positive")
negative_model_path_by_restaurant = os.path.join(w2v_models_path_by_restaurant, "negative")

positive_model_path_tf_itf_city = os.path.join(w2v_models_path_tf_itf_city, "positive")
negative_model_path_tf_itf_city = os.path.join(w2v_models_path_tf_itf_city, "negative")