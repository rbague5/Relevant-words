"""
Heatmap topic x restaurante x polaridad para SCCI + estadisticas para el texto del paper.

Salidas:
  images/figures_paper_restaurant/heatmap_scci_topics.pdf / .png
  images/figures_paper_restaurant/heatmap_scci_topics.csv   (formato largo, ya normalizado)
  Por consola: [X]/12 por topic, topics distintos medios por polaridad,
               % clusters sin topic (solo SCCI) y topics sin mapear.
"""
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import top_n_nearest_points

# ----------------------------------------------------------------------------- config
METRIC = "scci"
BASE = os.path.join("data_restaurants", "output", f"top_{top_n_nearest_points}")
OUT_DIR = os.path.join("images", "figures_paper_restaurant")
OUT_NAME = "heatmap_scci_topics"

RESTAURANTS = {
    "moscow": ["691623", "2419130", "7122712"],
    "madrid": ["1027419", "6367286", "17340009"],
    "istanbul": ["1010030", "1644684", "1749881"],
    "barcelona": ["1082035", "3179219", "11906544"],
}
POLARITIES = ["negative", "positive"]

MIN_RESTAURANTS = 2  # un topic se muestra si aparece en >= N restaurantes...
ALWAYS_SHOW = [  # ...o si es uno de los idiosincraticos citados en el texto
    "Payment and Transactions", "Accessibility", "Entertainment", "Safety", "Human Resources",
]

# Sinonimos -> topic canonico (claves en minusculas). Lo que no este aqui se queda con su nombre.
CANON = {
    "ambience": "Ambience", "ambiance": "Ambience", "ambience and atmosphere": "Ambience",
    "atmosphere": "Ambience", "seating and ambiance": "Ambience", "dining environment": "Ambience",
    "service experience": "Service Experience", "customer interaction": "Service Experience",
    "staff and customer interaction": "Service Experience",
    "staff and roles": "Staff & Roles", "personnel and roles": "Staff & Roles",
    "management": "Management & Operations", "operations": "Management & Operations",
    "cuisine": "Cuisine",
    "food quality": "Food Quality", "flavor and aroma": "Food Quality",
    "ingredients": "Ingredients", "ingredients and utensils": "Ingredients",
    "desserts and baking": "Desserts", "desserts and sweets": "Desserts",
    "beverage": "Beverages", "beverages": "Beverages",
    "pricing": "Pricing",
    "customer feedback": "Feedback & Advice", "feedback": "Feedback & Advice",
    "advice and recommendations": "Feedback & Advice", "advice and guidance": "Feedback & Advice",
    "customer demographics": "Customer Demographics",
    "time of day": "Time of Day", "dining time": "Time of Day", "meal times": "Time of Day",
    "time and occasion": "Time of Day",
    "occasion": "Occasion & Events", "celebration": "Occasion & Events",
    "events and celebrations": "Occasion & Events", "events and occasions": "Occasion & Events",
    "tourism": "Tourism & Travel", "travel and tourism": "Tourism & Travel",
    "travel and leisure": "Tourism & Travel",
    "location": "Location", "location and venue": "Location", "location and architecture": "Location",
    "venue and infrastructure": "Venue & Infrastructure",
    "dining experience": "Dining Experience", "experience": "Dining Experience",
    "measurement": "Measurement & Quantity", "quantity": "Measurement & Quantity",
}
# Los idiosincraticos de ALWAYS_SHOW se normalizan a si mismos
for _t in ALWAYS_SHOW:
    CANON.setdefault(_t.lower(), _t)

LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.+?)\s*\(Count:\s*(\d+)\)", re.MULTILINE)
RESP_RE = re.compile(r"\[\d+\]\s*(.*?):\s*(.*?)(?=\n*\[\d+\]|$)", re.DOTALL)


# ----------------------------------------------------------------------------- lectura
def parse_generation2_md(path: str):
    """Devuelve {topic_nivel_1: count}. El count de nivel 1 ya incluye sus subtopics."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    topics = defaultdict(int)
    for level, name, count in LINE_RE.findall(content):
        if level == "1":
            topics[name.strip()] += int(count)
    return topics


def cluster_stats(path: str):
    """(#clusters, #sin_topic) con la misma logica que load_jsonl1."""
    n_clusters, n_none = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            info = json.loads(line)
            n_clusters += 1
            resp = info["responses"]
            if resp == "None" or not RESP_RE.findall(resp):
                n_none += 1
    return n_clusters, n_none


def collect():
    rows, stats, unmapped = [], [], defaultdict(set)
    for city, rids in RESTAURANTS.items():
        for rid in rids:
            for pol in POLARITIES:
                folder = os.path.join(BASE, city, rid, f"{METRIC}_{pol}")
                topics = parse_generation2_md(os.path.join(folder, "generation_2.md"))
                n_cl, n_none = cluster_stats(os.path.join(folder, "generation_1.jsonl"))
                stats.append(dict(city=city, restaurant=rid, polarity=pol, clusters=n_cl, non_assigned=n_none))
                merged = defaultdict(int)
                for name, cnt in topics.items():
                    canon = CANON.get(name.lower())
                    if canon is None:
                        unmapped[name].add(rid)
                        canon = name
                    merged[canon] += cnt
                for canon, cnt in merged.items():
                    rows.append(dict(city=city, restaurant=rid, polarity=pol, topic=canon, count=cnt))
    return pd.DataFrame(rows), pd.DataFrame(stats), unmapped


# ----------------------------------------------------------------------------- seleccion y matrices
def select_topics(df: pd.DataFrame):
    per_topic = df.groupby("topic").agg(n_rest=("restaurant", "nunique"), total=("count", "sum"))
    keep = per_topic[(per_topic.n_rest >= MIN_RESTAURANTS) | (per_topic.index.isin(ALWAYS_SHOW))]
    return keep.sort_values(["n_rest", "total"], ascending=False).index.tolist()


def build_matrix(df: pd.DataFrame, pol: str, topics: list, columns: list):
    mat = np.zeros((len(topics), len(columns)), dtype=int)
    sub = df[df.polarity == pol]
    for _, r in sub.iterrows():
        if r.topic in topics:
            mat[topics.index(r.topic), columns.index(r.restaurant)] = r["count"]
    return mat


# ----------------------------------------------------------------------------- figura
def draw_panel(ax, mat, title, row_labels, col_labels, vmax, show_ylabels):
    n_rows, n_cols = mat.shape
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    im = ax.imshow(np.ma.masked_equal(mat, 0), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels if show_ylabels else [], fontsize=7)
    ax.tick_params(axis="both", length=0, pad=1)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#dddddd", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    for sep in (2.5, 5.5, 8.5):
        ax.axvline(sep, color="black", linewidth=0.8)

    for i in range(n_rows):
        for j in range(n_cols):
            if mat[i, j] > 0:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] > 0.6 * vmax else "black")

    for k, city in enumerate(RESTAURANTS):
        ax.text(k * 3 + 1, -0.8, city.title(), ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_title(title, fontsize=8, pad=14)
    return im


def make_figure(df: pd.DataFrame, topics: list, columns: list):
    mats = {pol: build_matrix(df, pol, topics, columns) for pol in POLARITIES}
    vmax = max(m.max() for m in mats.values())
    height = 0.2 * len(topics) + 1.4
    fig, axes = plt.subplots(1, 2, figsize=(7.2, height), constrained_layout=True)
    titles = {"negative": r"(a) Negative comments $C^{-}$", "positive": r"(b) Positive comments $C^{+}$"}
    im = None
    for k, pol in enumerate(POLARITIES):
        im = draw_panel(axes[k], mats[pol], titles[pol], topics, columns, vmax, show_ylabels=(k == 0))
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("# clusters (SCCI)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.locator = matplotlib.ticker.MaxNLocator(integer=True)
    cbar.update_ticks()

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"{OUT_NAME}.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{OUT_NAME}.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------- estadisticas para el texto
def print_stats(df: pd.DataFrame, stats: pd.DataFrame, unmapped: dict):
    n_total = sum(len(v) for v in RESTAURANTS.values())

    pres = df.groupby(["topic", "polarity"])["restaurant"].nunique().unstack(fill_value=0)
    pres["both"] = df.groupby("topic")["restaurant"].nunique()
    pres = pres.sort_values("both", ascending=False)
    print(f"\n=== Restaurantes (de {n_total}) donde aparece cada topic (SCCI) ===")
    print(pres.to_string())

    distinct = df.groupby(["restaurant", "polarity"])["topic"].nunique().unstack()
    print("\n=== Topics distintos (nivel 1, canonicos) por restaurante y polaridad ===")
    print(distinct.to_string())
    print("Media:", distinct.mean().round(2).to_dict(), " | Total (suma):", distinct.sum().to_dict())
    print("(Puede diferir del '# Diff. Topics' de las tablas, que cuenta subtopics.)")

    print("\n=== Clusters sin topic (solo SCCI) ===")
    by_pol = stats.groupby("polarity")[["clusters", "non_assigned"]].sum()
    by_pol["pct"] = (100 * by_pol.non_assigned / by_pol.clusters).round(1)
    print(by_pol.to_string())
    tot_c, tot_n = stats.clusters.sum(), stats.non_assigned.sum()
    print(f"Total SCCI: {tot_n}/{tot_c} ({100 * tot_n / tot_c:.1f}%)")

    if unmapped:
        print("\n=== Topics SIN mapear en CANON (revisa si son sinonimos) ===")
        for name, rids in sorted(unmapped.items(), key=lambda x: -len(x[1])):
            print(f"  {name}  ->  {len(rids)} restaurante(s)")


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    df, stats, unmapped = collect()
    columns = [rid for rids in RESTAURANTS.values() for rid in rids]
    topics = select_topics(df)

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, f"{OUT_NAME}.csv"), index=False)
    make_figure(df, topics, columns)
    print_stats(df, stats, unmapped)
    print(f"\nFigura guardada en {OUT_DIR}/{OUT_NAME}.pdf|png  ({len(topics)} topics mostrados)")