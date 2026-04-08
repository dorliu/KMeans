"""KMeans clustering on the Seeds dataset without external dependencies."""
from __future__ import annotations

import csv
import itertools
import math
import random
from pathlib import Path
from typing import List, Sequence, Tuple

DATA_PATH = Path(__file__).parent / "data" / "seeds.csv"


Vector = List[float]


def load_seeds() -> Tuple[List[Vector], List[str]]:
    features: List[Vector] = []
    labels: List[str] = []
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Seeds 数据集包含 7 个连续特征
            features.append([
                float(row["area"]),
                float(row["perimeter"]),
                float(row["compactness"]),
                float(row["kernel_length"]),
                float(row["kernel_width"]),
                float(row["asymmetry_coefficient"]),
                float(row["kernel_groove_length"]),
            ])
            # 保存品种标签用于后续参考评估
            labels.append(row["species"])
    return features, labels


def zscore_normalize(X: Sequence[Vector]) -> Tuple[List[Vector], Vector, Vector]:
    if not X:
        return [], [], []
    dim = len(X[0])
    means = [0.0] * dim
    stds = [0.0] * dim
    n = len(X)
    for vector in X:
        for i, value in enumerate(vector):
            means[i] += value
    means = [value / n for value in means]
    for vector in X:
        for i, value in enumerate(vector):
            stds[i] += (value - means[i]) ** 2
    stds = [math.sqrt(value / n) or 1.0 for value in stds]
    # 返回标准化后的特征，方便与原始特征进行双向转换
    normalized = [
        [(value - means[i]) / stds[i] for i, value in enumerate(vector)]
        for vector in X
    ]
    return normalized, means, stds


def euclidean(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def mean_vector(vectors: Sequence[Vector]) -> Vector:
    dim = len(vectors[0])
    sums = [0.0] * dim
    for vector in vectors:
        for i, value in enumerate(vector):
            sums[i] += value
    return [value / len(vectors) for value in sums]


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: int | None = None) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids: List[Vector] = []

    def fit(self, X: Sequence[Vector]) -> List[int]:
        rng = random.Random(self.random_state)
        # 初始化：随机抽样k个样本作为初始质心
        self.centroids = [vector[:] for vector in rng.sample(list(X), self.n_clusters)]
        labels = [0] * len(X)
        for _ in range(self.max_iter):
            moved = False
            # assignment
            for idx, vector in enumerate(X):
                distances = [euclidean(vector, center) for center in self.centroids]
                new_label = min(range(self.n_clusters), key=lambda c: distances[c])
                if labels[idx] != new_label:
                    labels[idx] = new_label
                    moved = True
            # update
            new_centroids: List[Vector] = []
            for cluster in range(self.n_clusters):
                members = [vector for vector, label in zip(X, labels) if label == cluster]
                if members:
                    new_centroids.append(mean_vector(members))
                else:
                    # 处理空簇：保持旧质心，避免崩溃
                    new_centroids.append(self.centroids[cluster])
            shift = max(euclidean(a, b) for a, b in zip(self.centroids, new_centroids))
            self.centroids = new_centroids
            if shift <= self.tol and not moved:
                break
        return labels

    def inertia(self, X: Sequence[Vector], labels: Sequence[int]) -> float:
        total = 0.0
        for vector, label in zip(X, labels):
            total += euclidean(vector, self.centroids[label]) ** 2
        return total


def silhouette_score(X: Sequence[Vector], labels: Sequence[int], n_clusters: int) -> float:
    # Avoid heavy numpy implementation; use straightforward loops.
    silhouettes: List[float] = []
    for idx, vector in enumerate(X):
        own_cluster = labels[idx]
        same_cluster = [
            other for j, other in enumerate(X)
            if labels[j] == own_cluster and j != idx
        ]
        if same_cluster:
            a = sum(euclidean(vector, other) for other in same_cluster) / len(same_cluster)
        else:
            a = 0.0
        b = math.inf
        for cluster in range(n_clusters):
            if cluster == own_cluster:
                continue
            other_members = [
                other for j, other in enumerate(X)
                if labels[j] == cluster
            ]
            if not other_members:
                continue
            distance = sum(euclidean(vector, other) for other in other_members) / len(other_members)
            b = min(b, distance)
        if math.isinf(b):
            silhouettes.append(0.0)
            continue
        silhouettes.append((b - a) / max(a, b))
    return sum(silhouettes) / len(silhouettes)


def best_accuracy(labels: Sequence[int], targets: Sequence[str]) -> float:
    classes = sorted(set(targets))
    mapping_options = itertools.permutations(range(len(classes)))
    best = 0.0
    for perm in mapping_options:
        mapping = {cluster: classes[idx] for cluster, idx in enumerate(perm)}
        correct = sum(1 for label, target in zip(labels, targets) if mapping[label] == target)
        best = max(best, correct / len(targets))
    return best


def main() -> None:
    X_raw, y = load_seeds()
    X, means, stds = zscore_normalize(X_raw)

    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit(X)

    inertia_value = model.inertia(X, labels)
    silhouette = silhouette_score(X, labels, n_clusters=3)
    accuracy = best_accuracy(labels, y)

    print("KMeans聚类（自实现）——Seeds数据集")
    print(f"样本量: {len(X)}，特征: {len(X[0])}")
    print(f"SSE (惯性): {inertia_value:.2f}")
    print(f"轮廓系数: {silhouette:.3f}")
    print(f"最优标签映射准确率(仅参考): {accuracy:.3f}")
    print("聚类中心（标准化空间）:")
    for idx, center in enumerate(model.centroids):
        formatted = ", ".join(f"{value:.3f}" for value in center)
        print(f"  簇 {idx}: [{formatted}]")


if __name__ == "__main__":
    main()
