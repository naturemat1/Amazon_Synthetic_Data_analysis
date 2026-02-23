"""
Data loading module for Amazon Sales Data Analysis.
Handles reading and initial parsing of the dataset.
"""

import pandas as pd
from typing import Optional
from pathlib import Path

from ..config.settings import DATASET, DTYPES


class DataLoader:
    """Handles loading and initial parsing of the Amazon dataset."""
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Parameters
        ----------
        filepath : str, optional
            Path to the CSV file. Defaults to DATASET from settings.
        """
        self.filepath = filepath or DATASET
        self._df: Optional[pd.DataFrame] = None
    
    def load(self, parse_dates: bool = True) -> pd.DataFrame:
        """
        Load the dataset from CSV.
        
        Parameters
        ----------
        parse_dates : bool
            Whether to parse the OrderDate column as datetime
        
        Returns
        -------
        pd.DataFrame
            The loaded dataframe
        """
        parse_dates_list = ["OrderDate"] if parse_dates else False
        
        self._df = pd.read_csv(
            self.filepath,
            parse_dates=parse_dates_list,
            dtype=DTYPES
        )
        
        return self._df
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the loaded dataframe."""
        if self._df is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self._df
    
    def get_info(self) -> None:
        """Print information about the loaded dataset."""
        if self._df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*80)
        print("DATASET INFORMATION")
        print("="*80)
        print(f"   Shape: {self._df.shape}")
        print(f"   Total Registros: {self._df.shape[0]:,}")
        print(f"   Total Variables: {self._df.shape[1]}")
        
        print("\nPrimeras y Ultimas 5 filas:")
        print(self._df.head())
        print(self._df.tail())
        
        print(f"\nTipos de Variables:")
        print(f"{self._df.info()}\n")
    
    def validate_file_exists(self) -> bool:
        """Check if the data file exists."""
        return Path(self.filepath).exists()


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load the dataset.
    
    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file
    
    Returns
    -------
    pd.DataFrame
        The loaded dataframe
    """
    loader = DataLoader(filepath)
    return loader.load()
