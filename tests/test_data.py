"""
Tests for data loading module.
"""

import pytest
import pandas as pd
from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_loader_initialization(self):
        """Test DataLoader can be initialized."""
        loader = DataLoader("test.csv")
        assert loader.filepath == "test.csv"
    
    def test_loader_default_path(self):
        """Test DataLoader uses default path."""
        loader = DataLoader()
        assert loader.filepath == "amazon.csv"
    
    def test_validate_file_exists(self):
        """Test file existence check."""
        loader = DataLoader("amazon.csv")
        assert loader.validate_file_exists() is True


def test_human_format():
    """Test human format function."""
    from src.core.formatters import human_format
    
    assert human_format(1000) == "1.0k"
    assert human_format(1000000) == "1.0M"
    assert human_format(500) == "500"


def test_config_settings():
    """Test configuration settings."""
    from src.config import settings
    
    assert settings.DATASET == "amazon.csv"
    assert "TotalAmount" in settings.NUMERIC_VARS
    assert settings.KMEANS_CONFIG["n_clusters"] == 4
