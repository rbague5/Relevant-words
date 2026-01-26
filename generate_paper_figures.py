import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
import numpy as np

# Load
def generate_plots(city, metric, topics, restaurant_id=None, algorithm="umap"):
    tsne_path = f"../results_by_city_paper_top20/{city}/results/plot_pkl_obj/"
    if restaurant_id is not None:
        tsne_path = f"../results_by_restaurant_paper_top20/{city}/{restaurant_id}/results/plot_pkl_obj/"

    cleaned_metric = metric.replace('-', '_')
    pattern = re.compile(rf".*{cleaned_metric}.*\.pkl$")

    pkl_files = [
        os.path.join(tsne_path, fname)
        for fname in os.listdir(tsne_path)
        if pattern.match(fname)
    ]
    if len(pkl_files) != 2:
        raise ValueError(f"Expected 2 files for metric {cleaned_metric}, found {len(pkl_files)}")

    for file in pkl_files:
        p = Path(file).name.replace('.pkl', '.png').replace(f'{algorithm}_inputs_', '')
        with open(file, "rb") as f:
            tsne_data = pickle.load(f)

        X_embedded = tsne_data["X_embedded"]
        words = tsne_data["words"]
        labels = tsne_data["labels"]
        n_samples = tsne_data["n_samples"]

        # Plot
        fig = plt.figure(figsize=(15, 10))  # FINAL image size
        ax = fig.add_axes([0.05, 0.08, 0.65, 0.84])  # FIXED plot area
        ax.set_facecolor("#f5f2f2")
        ax.grid(color='white', linestyle='-', linewidth=1)

        # Unique labels
        unique_labels = sorted(set(labels))

        labels_mapping = {}
        metric_key = re.sub(r'_\d+\.png$', '', p)
        topic_by_cluster = topics.get(metric_key).get('topics_details')
        for r in topic_by_cluster:
            if r.get("responses", None) is None:
                lab = "none"
            elif isinstance(r.get("responses", None), list):
                lab = r.get("responses", None)[-1]["topic"].lower()
                if isinstance(r.get("subtopics", None), list):
                    lab = r.get("subtopics", None)[-1].split(":")[0].lower()
            else:
                raise ValueError("Unexpected topic response format.")
            labels_mapping[r["id"]] = lab[0].upper() + lab[1:]

        n_labels = len(unique_labels)

        # Generate enough distinct colors using a large palette
        palette = sns.color_palette("tab20", n_colors=n_labels)  # tab20 can go up to 20, adjust if more labels
        if n_labels > 20:
            # fallback: generate evenly spaced colors in HSV
            import matplotlib.colors as mcolors
            palette = [mcolors.hsv_to_rgb((i/n_labels, 0.7, 0.9)) for i in range(n_labels)]
        random.shuffle(palette)
        # Generate enough marker styles
        marker_styles = ['o','s','^','v','<','>','D','d','P','X','*','h','H','p','+','x','|','_','1','2','3','4']
        if n_labels > len(marker_styles):
            # extend markers by repeating shapes with different rotations or size adjustments
            extra_needed = n_labels - len(marker_styles)
            marker_styles += ['o'] * extra_needed  # fallback: reuse 'o' (still unique via color)
        marker_map = {lbl: marker_styles[i % len(marker_styles)] for i, lbl in enumerate(unique_labels)}

        # Scatter plot
        for lbl in unique_labels:
            idx = np.array(labels[:n_samples]) == lbl
            ax.scatter(
                X_embedded[idx, 0],
                X_embedded[idx, 1],
                label=lbl,
                c=[palette[unique_labels.index(lbl)]],
                marker=marker_map[lbl],
                s=40,
                edgecolor='w',
                linewidth=0.5
            )

        # Annotate
        for label, x, y in zip(words, X_embedded[:, 0], X_embedded[:, 1]):
            ax.annotate(label, xy=(x, y), xytext=(0, 0), fontsize=15, textcoords="offset points")

        # Dynamic legend
        handles = [
            Line2D([0], [0], marker=marker_map[lbl], color='w',
                   label=labels_mapping[lbl], markerfacecolor=palette[unique_labels.index(lbl)], markersize=10)
            for lbl in unique_labels
        ]

        legend = ax.legend(
            handles=handles,
            title="Topics",
            loc="upper left",
            bbox_to_anchor=(0.73, 0.92),
            bbox_transform=fig.transFigure,
            frameon=True,
            borderpad=0.6,
            labelspacing=0.6,
            handletextpad=0.6
        )

        # plt.tight_layout()
        fig.subplots_adjust(right=0.75)
        output_dir = f"../figures_paper_city/{city}"
        if restaurant_id is not None:
            output_dir = f"../figures_paper_restaurant/{city}/{restaurant_id}"

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            f"{output_dir}/{p}",
            facecolor=plt.gcf().get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.05
        )
        plt.close()
