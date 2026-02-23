"""
Report generation module for Amazon Sales Data Analysis.
"""

import pandas as pd
from typing import Optional


class ReportGenerator:
    """Generates summary reports for the analysis."""
    
    def __init__(self, df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None):
        """
        Initialize the ReportGenerator.
        
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataframe
        original_df : pd.DataFrame
            Original dataframe
        """
        self._df = df
        self._original_df = original_df if original_df is not None else df
    
    def print_data_info(self) -> None:
        """Print basic data information."""
        print("\n" + "="*80)
        print("1. DIMENSIONES ORIGINALES - DATASET:")
        print("="*80)
        print(f"   Shape: {self._df.shape}")
        print(f"   Total Registros: {self._df.shape[0]:,}")
        print(f"   Total Variables: {self._df.shape[1]}")
        
        print("\n1.1 Primeras y Ultimas 5 filas:")
        print(self._df.head())
        print(self._df.tail())
        
        print(f"\n1.2 Tipos de Variables")
        print(f"{self._df.info()}\n")
    
    def print_numeric_stats(self) -> None:
        """Print numeric statistics."""
        print("\n" + "="*80)
        print("2.1 ESTADÍSTICAS BÁSICAS - Variables Numéricas:")
        print("="*80)
        print(self._df.describe())
    
    def print_categorical_stats(self) -> None:
        """Print categorical statistics."""
        print("\n" + "="*80)
        print("2.2 ESTADÍSTICAS BÁSICAS - Variables Categóricas:")
        print("="*80)
        print(self._df.describe(include=["category", "string"]).T)
    
    def print_top_products_quantity(self, n: int = 5) -> pd.DataFrame:
        """Print top products by quantity."""
        print("\n" + "="*80)
        print(f"14. TOP {n} Productos más vendidos")
        print("="*80)
        
        result = (
            self._original_df.groupby("ProductName")
            .agg(
                UnidadesVendidas=("Quantity", "sum"),
                IngresoTotal=("TotalAmount", "sum")
            )
            .sort_values("UnidadesVendidas", ascending=False)
            .head(n)
        )
        print(result)
        return result
    
    def print_top_products_income(self, n: int = 5) -> pd.DataFrame:
        """Print top products by income."""
        print("\n" + "="*80)
        print(f"15. TOP {n} Productos que generaron más ingresos")
        print("="*80)
        
        result = (
            self._original_df.groupby("ProductName")
            .agg(
                UnidadesVendidas=("Quantity", "sum"),
                IngresoTotal=("TotalAmount", "sum")
            )
            .sort_values("IngresoTotal", ascending=False)
            .head(n)
        )
        print(result)
        return result
    
    def print_top_customers(self, n: int = 5) -> pd.DataFrame:
        """Print top customers."""
        print("\n" + "="*80)
        print(f"16. TOP {n} Clientes que generaron más ingresos")
        print("="*80)
        
        result = (
            self._original_df.groupby(["CustomerID", "CustomerName"])
            .agg(
                TotalCompras=("Quantity", "sum"),
                IngresoTotal=("TotalAmount", "sum")
            )
            .sort_values("IngresoTotal", ascending=False)
            .head(n)
        )
        print(result)
        return result
    
    def generate_full_report(self) -> None:
        """Generate a complete summary report."""
        self.print_data_info()
        self.print_numeric_stats()
        self.print_categorical_stats()
        self.print_top_products_quantity()
        self.print_top_products_income()
        self.print_top_customers()


def generate_summary(df: pd.DataFrame, original_df: pd.DataFrame) -> ReportGenerator:
    """
    Convenience function to create a ReportGenerator.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe
    original_df : pd.DataFrame
        Original dataframe
    
    Returns
    -------
    ReportGenerator
    """
    return ReportGenerator(df, original_df)
