import os


def load_spacy():
    import spacy
    # spaCy config
    spacy_model = "en_core_web_lg"
    spacy.cli.download(spacy_model)
    nlp = spacy.load(spacy_model, disable=["ner", "parser"])
    return nlp


def load_nltk():
    import nltk
    # NLTK config
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('words')
    nltk.download('wordnet')


columns_to_delete = ["images", "index", "language", "title", "url", "date"]

base_path = "./dataset"
source_path = os.path.join(base_path, "raw")
target_path = os.path.join(base_path, "input")

cities = ["gijon", "moscow", "madrid", "istanbul", "barcelona"]
rating_threshold = 40
top_n_restaurants = 3
top_n_nearest_points = 10
n_clusters_range = range(3, 15)
embedding_model_name = "e5" # "fasttext", "google", "word2vec", "glove", "minilm", "sbert", "e5"

w2v_models_path_by_city = f"./models/"
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
