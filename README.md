# KMeans
# KMeans Clustering from Scratch & PCA Visualization
基于 Seeds 数据集的 K-Means 聚类与主成分分析 (PCA) 可视化

## 📖 Introduction
[cite_start]This repository contains a complete unsupervised learning pipeline utilizing the UCI Seeds dataset (210 samples across 3 wheat varieties). [cite_start]The project aims to evaluate KMeans clustering feasibility using both `scikit-learn` and a pure Python implementation built from scratch.

## 🛠️ Tech Stack
* Python 3.x
* [cite_start]`scikit-learn` (for benchmark and PCA) 
* [cite_start]`numpy` (for matrix operations) 
* [cite_start]`matplotlib` (for visualization) 

## ✨ Core Highlights
* [cite_start]**Implementation from Scratch**: Includes `kmeans_fromscratch.py`, a pure Python implementation of the KMeans algorithm without relying on heavy machine learning libraries.
* [cite_start]**Data Preprocessing**: Applied Z-score standardization (`StandardScaler`) to eliminate the dominance of large-scale features.
* [cite_start]**Dimensionality Reduction**: Utilized PCA to map 7-dimensional features into a 2D plane for clear cluster visualization.
* [cite_start]**Quantitative Evaluation**: Comprehensive evaluation using Inertia (SSE), Silhouette Score, and optimal label mapping accuracy.



## 🚀 Simulation Performance
* [cite_start]**Inertia (SSE)**: 430.66 
* [cite_start]**Silhouette Score**: 0.401 
* [cite_start]**Mapping Accuracy**: 91.9% 
[cite_start]*(Note: Optimal cluster number identified as k=3 via the Elbow method )*

## 📄 Full Report
For algorithm details and analysis, please refer to the uploaded `KMeans_Clustering_Report.pdf`.
