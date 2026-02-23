"""Clustering package for Amazon Sales Data Analysis.

This package handles PCA, K-Means, and customer segmentation.
"""

from .pca import PCAModel, perform_pca
from .kmeans import KMeansModel, perform_kmeans
from .segmentation import CustomerSegmenter, segment_customers

__all__ = [
    "PCAModel",
    "perform_pca",
    "KMeansModel",
    "perform_kmeans",
    "CustomerSegmenter",
    "segment_customers",
]
