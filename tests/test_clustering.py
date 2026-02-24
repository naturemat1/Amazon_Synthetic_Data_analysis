"""
Tests for clustering module.
"""

import pytest
import numpy as np
import pandas as pd
from src.clustering.pca import PCAModel
from src.clustering.kmeans import KMeansModel


class TestPCAModel:
    """Test cases for PCAModel class."""
    
    def test_pca_initialization(self):
        """Test PCA can be initialized."""
        pca = PCAModel(n_components=2)
        assert pca.n_components == 2
    
    def test_pca_fit_transform(self):
        """Test PCA fit and transform."""
        # Create sample data
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'c': [1, 1, 1, 1, 1]
        })
        
        pca = PCAModel(n_components=2)
        result = pca.fit_transform(data)
        
        assert result.shape == (5, 2)
        assert pca.total_explained_variance > 0
    
    def test_get_loadings(self):
        """Test PCA loadings extraction."""
        data = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10]
        })
        
        pca = PCAModel(n_components=2)
        pca.fit_transform(data)
        
        loadings = pca.get_loadings(['a', 'b'])
        assert loadings.shape == (2, 2)


class TestKMeansModel:
    """Test cases for KMeansModel class."""
    
    def test_kmeans_initialization(self):
        """Test KMeans can be initialized."""
        kmeans = KMeansModel(n_clusters=3)
        assert kmeans.n_clusters == 3
    
    def test_kmeans_fit(self):
        """Test KMeans fit."""
        # Create sample data
        X = np.array([
            [1, 2],
            [1, 4],
            [1, 0],
            [10, 2],
            [10, 4],
            [10, 0]
        ])
        
        kmeans = KMeansModel(n_clusters=2)
        labels = kmeans.fit_predict(X)
        
        assert len(labels) == 6
        assert set(labels) <= {0, 1}
    
    def test_inertia_calculation(self):
        """Test inertia calculation."""
        X = np.array([
            [1, 2],
            [1, 4],
            [10, 2],
            [10, 4]
        ])
        
        kmeans = KMeansModel(n_clusters=2)
        kmeans.fit(X)
        
        assert kmeans.inertia >= 0
