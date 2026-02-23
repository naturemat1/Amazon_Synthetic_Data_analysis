"""
K-Means clustering module for Amazon Sales Data Analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt

from ..config.settings import KMEANS_CONFIG
from ..core.formatters import get_formatter


class KMeansModel:
    """Handles K-Means clustering operations."""
    
    def __init__(self, n_clusters: Optional[int] = None, **kwargs):
        """
        Initialize K-Means model.
        
        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters. Defaults to KMEANS_CONFIG.
        **kwargs
            Additional K-Means parameters
        """
        config = {**KMEANS_CONFIG, **kwargs}
        self.n_clusters = n_clusters or config["n_clusters"]
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=config.get("init", "k-means++"),
            n_init=config.get("n_init", 50),
            max_iter=config.get("max_iter", 100),
            algorithm=config.get("algorithm", "lloyd"),
            random_state=config.get("random_state", 42)
        )
        
        self._labels: Optional[np.ndarray] = None
        self._inertia: Optional[float] = None
        self._centroids: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> 'KMeansModel':
        """
        Fit K-Means model to data.
        
        Parameters
        ----------
        X : np.ndarray
            Data to cluster
        
        Returns
        -------
        self
        """
        self.kmeans.fit(X)
        self._labels = self.kmeans.labels_
        self._inertia = self.kmeans.inertia_
        self._centroids = self.kmeans.cluster_centers_
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and return cluster labels.
        
        Parameters
        ----------
        X : np.ndarray
            Data to cluster
        
        Returns
        -------
        np.ndarray
            Cluster labels
        """
        self.fit(X)
        return self._labels
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Parameters
        ----------
        X : np.ndarray
            Data to predict
        
        Returns
        -------
        np.ndarray
            Cluster labels
        """
        return self.kmeans.predict(X)
    
    @property
    def labels(self) -> np.ndarray:
        """Get cluster labels."""
        return self._labels
    
    @property
    def inertia(self) -> float:
        """Get inertia (within-cluster sum of squares)."""
        return self._inertia
    
    @property
    def centroids(self) -> np.ndarray:
        """Get cluster centroids."""
        return self._centroids
    
    def calculate_elbow_method(self, X: np.ndarray, 
                                k_range: range = range(2, 11)) -> Tuple[List[int], List[float]]:
        """
        Perform elbow method to find optimal k.
        
        Parameters
        ----------
        X : np.ndarray
            Data to cluster
        k_range : range
            Range of k values to try
        
        Returns
        -------
        tuple
            (k_values, inertia_values)
        """
        print("\n" + "="*80)
        print("MÉTODO DEL CODO JAMBÚ")
        print("="*80)
        
        inertias = []
        
        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                n_init=50,
                max_iter=100,
                algorithm='lloyd',
                random_state=42
            )
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertias, marker='o', linestyle='--')
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Inercia")
        plt.title("Método del Codo (Clientes - ACP)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return list(k_range), inertias
    
    def plot_elbow(self, k_range: List[int], inertias: List[float]) -> None:
        """
        Plot elbow method results.
        
        Parameters
        ----------
        k_range : list
            k values
        inertias : list
            Inertia values
        """
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, inertias, marker='o', linestyle='--')
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Inercia")
        plt.title("Método del Codo (Clientes - ACP)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def calculate_inertia_analysis(self, X: np.ndarray) -> dict:
        """
        Calculate inertia analysis.
        
        Parameters
        ----------
        X : np.ndarray
            Data used for clustering
        
        Returns
        -------
        dict
            Inertia analysis results
        """
        inertia_intra = self._inertia
        global_center = X.mean(axis=0)
        inertia_total = np.sum((X - global_center) ** 2)
        inertia_inter = inertia_total - inertia_intra
        
        return {
            'inertia_total': inertia_total,
            'inertia_intra': inertia_intra,
            'inertia_inter': inertia_inter,
            'pct_intra': 100 * inertia_intra / inertia_total,
            'pct_inter': 100 * inertia_inter / inertia_total,
        }
    
    def print_inertia_analysis(self, X: np.ndarray) -> None:
        """
        Print inertia analysis results.
        
        Parameters
        ----------
        X : np.ndarray
            Data used for clustering
        """
        analysis = self.calculate_inertia_analysis(X)
        
        print("\n" + "="*80)
        print("ANÁLISIS DE INERCIAS")
        print("="*80)
        print(f"Inercia total:      {analysis['inertia_total']:.4f}")
        print(f"Inercia intraclase: {analysis['inertia_intra']:.4f} ({analysis['pct_intra']:.2f}%)")
        print(f"Inercia interclase: {analysis['inertia_inter']:.4f} ({analysis['pct_inter']:.2f}%)")


def perform_kmeans(X: np.ndarray, n_clusters: Optional[int] = None) -> Tuple[KMeansModel, np.ndarray]:
    """
    Convenience function to perform K-Means clustering.
    
    Parameters
    ----------
    X : np.ndarray
        Data to cluster
    n_clusters : int, optional
        Number of clusters
    
    Returns
    -------
    tuple
        (KMeansModel, labels)
    """
    model = KMeansModel(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return model, labels
