"""Generate a scatter plot for KMeans clustering on the Seeds dataset."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "seeds.csv"


def load_seeds() -> tuple[np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[str] = []
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
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
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    y = np.array([label_to_idx[label] for label in labels])
    return np.array(features, dtype=float), y


def main() -> None:
    X, y = load_seeds()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 4))

    ax1 = plt.subplot(1, 2, 1)
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k")
    plt.title("Seeds True Labels (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(*scatter1.legend_elements(), title="Species", loc="lower left", fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", edgecolor="k")
    plt.title("KMeans Cluster Labels (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend(*scatter2.legend_elements(), title="Cluster", loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig("kmeans_result.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
