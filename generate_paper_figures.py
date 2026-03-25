import os
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pickle
import numpy as np
import textwrap


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

        # Plot with FIXED dimensions and minimal margins
        fig = plt.figure(figsize=(15, 10))

        # Remove all automatic spacing
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # FIXED plot area with 0.05 margins on all sides
        # Legend takes fixed space on the right
        legend_width = 0.10  # Fixed width for legend area
        margin = 0.05

        # Calculate plot area: left margin + plot width + legend width + right margin = 1.0
        plot_width = 1.0 - (2 * margin) - legend_width
        plot_height = 1.0 - (2 * margin)

        ax = fig.add_axes([margin, margin, plot_width, plot_height])  # [left, bottom, width, height]
        ax.set_facecolor("#f5f2f2")
        ax.grid(color='white', linestyle='-', linewidth=1)

        # Set fixed axis limits to ensure consistent plot area
        ax.set_xlim(X_embedded[:, 0].min() - 1, X_embedded[:, 0].max() + 1)
        ax.set_ylim(X_embedded[:, 1].min() - 1, X_embedded[:, 1].max() + 1)

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
        palette = sns.color_palette("tab20", n_colors=n_labels)
        if n_labels > 20:
            import matplotlib.colors as mcolors
            palette = [mcolors.hsv_to_rgb((i / n_labels, 0.7, 0.9)) for i in range(n_labels)]
        random.shuffle(palette)

        # Generate enough marker styles
        marker_styles = ['o', 's', '^', 'v', '<', '>', 'D', 'd', 'P', 'X', '*', 'h', 'H', 'p', '+', 'x', '|', '_', '1',
                         '2', '3', '4']
        if n_labels > len(marker_styles):
            extra_needed = n_labels - len(marker_styles)
            marker_styles += ['o'] * extra_needed
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

        # Wrap long labels for multi-line display
        max_label_width = 20  # Characters per line - adjust as needed
        wrapped_labels = {}
        for lbl in unique_labels:
            original_label = labels_mapping[lbl]
            wrapped_labels[lbl] = '\n'.join(textwrap.wrap(original_label, width=max_label_width))

        # Create legend handles with wrapped labels
        handles = [
            Line2D([0], [0], marker=marker_map[lbl], color='w',
                   label=wrapped_labels[lbl],
                   markerfacecolor=palette[unique_labels.index(lbl)],
                   markersize=10)
            for lbl in unique_labels
        ]

        # Create legend with FIXED size box positioned in the legend area
        legend = ax.legend(
            handles=handles,
            title="Topics",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),  # Just outside the plot area
            bbox_transform=ax.transAxes,
            frameon=True,
            borderpad=0.8,
            labelspacing=1.0,
            handletextpad=0.6,
            fontsize=9,
            title_fontsize=11,
            ncol=1,
            fancybox=True,
            shadow=False,
            framealpha=1.0,
            # edgecolor='black',
            # facecolor='white',
            alignment = 'left'
        )

        # Force legend to have a fixed width
        legend_box = legend.get_frame()
        legend_box.set_linewidth(1.5)
        legend._legend_box.set_width(150)  # pixels - adjust as needed

        output_dir = f"../figures_paper_city/{city}"
        if restaurant_id is not None:
            output_dir = f"../figures_paper_restaurant/{city}/{restaurant_id}"

        os.makedirs(output_dir, exist_ok=True)
        # plt.tight_layout()
        # Save with FIXED dimensions - NO extra padding
        plt.savefig(
            f"{output_dir}/{p}",
            facecolor=plt.gcf().get_facecolor(),
            bbox_inches=None,  # Don't adjust bounds
            dpi=100,  # Fixed DPI
            pad_inches=0  # NO extra padding
        )
        plt.close()