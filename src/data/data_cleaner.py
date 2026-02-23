"""
Data cleaning and preprocessing module for Amazon Sales Data Analysis.
Handles duplicate removal, missing values, and column transformations.
"""

import pandas as pd
import numpy as np
from typing import Optional, List

from ..config.settings import COLS_TO_DROP


class DataCleaner:
    """Handles data cleaning and preprocessing operations."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataCleaner with a dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to clean
        """
        self._df = df.copy()
        self._original_df = df.copy()
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the cleaned dataframe."""
        return self._df
    
    def remove_duplicates(self) -> int:
        """
        Remove duplicate rows from the dataframe.
        
        Returns
        -------
        int
            Number of duplicates removed
        """
        duplicates = self._df.duplicated().sum()
        
        print("\n" + "="*80)
        print("ELIMINAR REGISTROS DUPLICADOS:")
        print("="*80)
        print(f"   Filas Duplicadas: {duplicates}")
        
        if duplicates > 0:
            self._df = self._df.drop_duplicates()
            print(f"   {duplicates} filas duplicadas eliminadas")
        else:
            print("No se encontro filas duplicadas")
        
        return duplicates
    
    def convert_date_column(self, column: str = "OrderDate") -> None:
        """
        Convert a column to datetime format.
        
        Parameters
        ----------
        column : str
            Name of the column to convert
        """
        print("\n" + "="*80)
        print(f"CONVERSIÓN DE FECHA DE STRING A {column}:")
        print("="*80)
        
        self._df[column] = pd.to_datetime(self._df[column], errors='coerce')
    
    def drop_columns(self, columns: Optional[List[str]] = None) -> List[str]:
        """
        Remove irrelevant columns from the dataframe.
        
        Parameters
        ----------
        columns : list, optional
            List of columns to drop. Defaults to COLS_TO_DROP.
        
        Returns
        -------
        list
            List of columns that were dropped
        """
        columns = columns or COLS_TO_DROP
        
        print("\n" + "="*80)
        print("ELIMINAR VARIABLES IRRELEVANTES")
        print("="*80)
        print(f"Variables eliminadas: {columns}")
        
        existing_cols = [c for c in columns if c in self._df.columns]
        self._df = self._df.drop(columns=existing_cols, errors='ignore')
        
        print("Variables restantes:", self._df.columns.tolist())
        
        return existing_cols
    
    def check_missing_values(self) -> pd.DataFrame:
        """
        Check for missing values in the dataframe.
        
        Returns
        -------
        pd.DataFrame
            Summary of missing values
        """
        print("\n" + "="*80)
        print("REVISIÓN DE VALORES NULOS")
        print("="*80)
        
        missing_data = self._df.isnull().sum()
        missing_percentage = (missing_data / len(self._df)) * 100
        
        missing_summary = pd.DataFrame({
            'Cantidad Nulos': missing_data,
            'Porcentaje': missing_percentage
        })
        
        missing_df = missing_summary[missing_summary['Cantidad Nulos'] > 0]
        
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print(f"   Valores nulos encontrados: {len(missing_df)}")
        
        return missing_df
    
    def get_total_income(self, column: str = "TotalAmount") -> float:
        """
        Calculate the total income from the dataset.
        
        Parameters
        ----------
        column : str
            Column to sum
        
        Returns
        -------
        float
            Total sum
        """
        print("\n" + "="*80)
        print(f"7. Ingreso Total")
        print("="*80)
        
        total_income = self._original_df[column].sum()
        
        # Format as currency (Latin America style)
        formatted = f"{total_income:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        print(formatted)
        
        return total_income
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Return the fully processed dataframe ready for analysis.
        
        Returns
        -------
        pd.DataFrame
            The cleaned and processed dataframe
        """
        return self._df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to clean the entire dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        The raw dataframe
    
    Returns
    -------
    pd.DataFrame
        The cleaned dataframe
    """
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates()
    cleaner.convert_date_column()
    cleaner.drop_columns()
    cleaner.check_missing_values()
    cleaner.get_total_income()
    
    return cleaner.get_processed_data()
