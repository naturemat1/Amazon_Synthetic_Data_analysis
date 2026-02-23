"""
PCA (Principal Component Analysis) module for Amazon Sales Data Analysis.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import matplotlib.pyplot as plt

from ..config.settings import PCA_CONFIG
from ..core.formatters import get_formatter


class PCAModel:
    """Handles PCA analysis for dimensionality reduction."""
    
    def __init__(self, n_components: Optional[int] = None, random_state: Optional[int] = None):
        """
        Initialize PCA model.
        
        Parameters
        ----------
        n_components : int, optional
            Number of components. Defaults to PCA_CONFIG.
        random_state : int, optional
            Random state. Defaults to PCA_CONFIG.
        """
        self.n_components = n_components or PCA_CONFIG["n_components"]
        self.random_state = random_state or PCA_CONFIG["random_state"]
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.scaler = StandardScaler()
        self._X_scaled: Optional[np.ndarray] = None
        self._X_pca: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit scaler and PCA, then transform data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to transform
        
        Returns
        -------
        np.ndarray
            PCA-transformed data
        """
        self._X_scaled = self.scaler.fit_transform(data.fillna(0))
        self._X_pca = self.pca.fit_transform(self._X_scaled)
        self._explained_variance = self.pca.explained_variance_ratio_
        
        return self._X_pca
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted scaler and PCA.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to transform
        
        Returns
        -------
        np.ndarray
            PCA-transformed data
        """
        X_scaled = self.scaler.transform(data.fillna(0))
        return self.pca.transform(X_scaled)
    
    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        return self._explained_variance
    
    @property
    def total_explained_variance(self) -> float:
        """Get total explained variance."""
        return self._explained_variance.sum()
    
    def get_loadings(self, feature_names: list) -> pd.DataFrame:
        """
        Get PCA loadings (component weights).
        
        Parameters
        ----------
        feature_names : list
            Names of original features
        
        Returns
        -------
        pd.DataFrame
            Loadings dataframe
        """
        columns = [f'PC{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(
            self.pca.components_.T,
            columns=columns,
            index=feature_names
        )
    
    def plot_pca_scatter(self, labels: Optional[np.ndarray] = None,
                         title: str = "Proyección ACP de Clientes",
                         figsize: Tuple[int, int] = (7, 6)) -> None:
        """
        Plot PCA scatter plot.
        
        Parameters
        ----------
        labels : np.ndarray, optional
            Cluster labels for coloring
        title : str
            Plot title
        figsize : tuple
            Figure size
        """
        if self._X_pca is None:
            raise ValueError("No PCA transformation done. Call fit_transform first.")
        
        plt.figure(figsize=figsize)
        
        if labels is not None:
            scatter = plt.scatter(
                self._X_pca[:, 0], 
                self._X_pca[:, 1], 
                c=labels, 
                alpha=0.6,
                cmap='tab10'
            )
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(self._X_pca[:, 0], self._X_pca[:, 1], alpha=0.6, color='gray')
        
        plt.xlabel(f"PC1 ({self._explained_variance[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({self._explained_variance[1]*100:.1f}%)")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def print_summary(self) -> None:
        """Print PCA summary."""
        print("\n" + "="*80)
        print("ACP")
        print("="*80)
        print(f"\nVarianza explicada PC1 + PC2: {self.total_explained_variance:.2%}")


def perform_pca(data: pd.DataFrame, 
                n_components: Optional[int] = None) -> Tuple[PCAModel, np.ndarray]:
    """
    Convenience function to perform PCA.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to transform
    n_components : int, optional
        Number of components
    
    Returns
    -------
    tuple
        (PCAModel, transformed_data)
    """
    model = PCAModel(n_components=n_components)
    X_pca = model.fit_transform(data)
    return model, X_pca
