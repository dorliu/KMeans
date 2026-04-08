"""Generate additional visualizations for KMeans on the Seeds dataset."""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path(__file__).parent


def load_data():
    features: list[list[float]] = []
    labels: list[str] = []
    with (OUTPUT_DIR / "data" / "seeds.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append([
                float(row["area"]),
                float(row["perimeter"]),
                float(row["compactness"]),
                float(row["kernel_length"]),
                float(row["kernel_width"]),
                float(row["asymmetry_coefficient"]),
                float(row["kernel_groove_length"]),
            ])
            labels.append(row["species"])
    feature_names = [
        "area",
        "perimeter",
        "compactness",
        "kernel_length",
        "kernel_width",
        "asymmetry_coeff",
        "groove_length",
    ]
    X = np.array(features, dtype=float)
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    y = np.array([label_to_idx[label] for label in labels])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, y, feature_names


def plot_feature_pairs(X, y, feature_names, max_plots: int = 6):
    pairs = list(itertools.combinations(range(X.shape[1]), 2))[:max_plots]
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle("Seeds Feature Pairs (Colored by Species)")
    scatter_handle = None
    for ax, (i, j) in zip(axes.ravel(), pairs):
        scatter_handle = ax.scatter(
            X[:, i], X[:, j], c=y, cmap="viridis", edgecolor="k", s=25
        )
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.grid(alpha=0.2)
    handles, labels = scatter_handle.legend_elements()
    fig.legend(handles, labels, title="Species", loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    output_path = OUTPUT_DIR / "feature_pairs.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_elbow(X_scaled):
    ks = range(1, 10)
    inertias = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X_scaled)
        inertias.append(model.inertia_)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.title("KMeans Elbow Curve (Seeds)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (SSE)")
    plt.grid(alpha=0.3)
    plt.xticks(list(ks))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_elbow.png", dpi=160)
    plt.close()


def plot_silhouette(X_scaled):
    ks = range(2, 10)
    scores = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
    plt.figure(figsize=(6, 4))
    plt.plot(list(ks), scores, marker="o", color="#ff7f0e")
    plt.title("KMeans Silhouette Scores (Seeds)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.grid(alpha=0.3)
    plt.xticks(list(ks))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_silhouette.png", dpi=160)
    plt.close()


def main() -> None:
    X, X_scaled, y, feature_names = load_data()
    plot_feature_pairs(X, y, feature_names)
    plot_elbow(X_scaled)
    plot_silhouette(X_scaled)
    print("Saved feature_pairs.png, kmeans_elbow.png, kmeans_silhouette.png")


if __name__ == "__main__":
    main()
