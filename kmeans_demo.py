"""KMeans clustering demonstration on the Seeds dataset."""
from __future__ import annotations

from itertools import permutations
from pathlib import Path
import csv

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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


def cluster_seeds(random_state: int = 42) -> dict:
    X, y = load_seeds()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=random_state)
    kmeans.fit(X_scaled)

    labels = kmeans.labels_
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    # Evaluate clustering by finding best label mapping (for reference only)
    best_accuracy = 0.0
    for perm in permutations(range(3)):
        mapped = np.array([perm[label] for label in labels])
        accuracy = (mapped == y).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy

    return {
        "inertia": inertia,
        "silhouette": silhouette,
        "centers": kmeans.cluster_centers_,
        "best_accuracy": best_accuracy,
    }


def main() -> None:
    result = cluster_seeds()
    print("KMeans on Seeds dataset")
    print(f"Inertia (SSE): {result['inertia']:.2f}")
    print(f"Silhouette score: {result['silhouette']:.3f}")
    print(f"Reference accuracy (best mapping): {result['best_accuracy']:.3f}")
    print("Cluster centers (standardized feature space):")
    for idx, center in enumerate(result["centers"]):
        formatted = ", ".join(f"{value:.3f}" for value in center)
        print(f"  Cluster {idx}: [{formatted}]")


if __name__ == "__main__":
    main()
