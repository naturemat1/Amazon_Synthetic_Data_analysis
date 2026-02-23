"""
Exploratory Data Analysis module for Amazon Sales Data Analysis.
Handles statistical analysis and data summaries.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

from ..config.settings import NUMERIC_VARS, CATEGORICAL_VARS, ID_VARS


class ExploratoryDataAnalysis:
    """Handles exploratory data analysis operations."""
    
    def __init__(self, df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None):
        """
        Initialize the EDA with a dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            The cleaned dataframe for analysis
        original_df : pd.DataFrame, optional
            The original dataframe (for joins, etc.)
        """
        self._df = df
        self._original_df = original_df if original_df is not None else df
    
    @property
    def data(self) -> pd.DataFrame:
        """Get the dataframe."""
        return self._df
    
    def basic_statistics_numeric(self) -> pd.DataFrame:
        """Calculate and print basic statistics for numeric variables."""
        print("\n" + "="*80)
        print("ESTADÍSTICAS BÁSICAS - VARIABLES NUMÉRICAS")
        print("="*80)
        stats = self._df.describe()
        print(stats)
        return stats
    
    def basic_statistics_categorical(self) -> pd.DataFrame:
        """Calculate and print basic statistics for categorical variables."""
        print("\n" + "="*80)
        print("ESTADÍSTICAS BÁSICAS - VARIABLES CATEGÓRICAS")
        print("="*80)
        stats = self._df.describe(include=["category", "string"]).T
        print(stats)
        return stats
    
    def get_top_products_by_quantity(self, n: int = 5) -> pd.DataFrame:
        """Get top N products by quantity sold."""
        print("\n" + "="*80)
        print(f"TOP {n} PRODUCTOS MÁS VENDIDOS")
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
    
    def get_top_products_by_income(self, n: int = 5) -> pd.DataFrame:
        """Get top N products by total income."""
        print("\n" + "="*80)
        print(f"TOP {n} PRODUCTOS CON MÁS INGRESOS")
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
    
    def get_top_customers_by_income(self, n: int = 5) -> pd.DataFrame:
        """Get top N customers by total income."""
        print("\n" + "="*80)
        print(f"TOP {n} CLIENTES CON MÁS INGRESOS")
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
    
    def get_correlation_matrix(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate correlation matrix for numeric columns."""
        
        df_numeric = self._df.select_dtypes(include=['number'])
        
        if columns:
            df_numeric = df_numeric[columns]
        
        corr_matrix = df_numeric.corr()
        print(corr_matrix.round(3))
        
        return corr_matrix
    
    def get_monthly_series(self) -> pd.Series:
        """Get monthly revenue time series."""
        return self._df.set_index('OrderDate')['TotalAmount'].resample('ME').sum()
    
    def get_id_analysis(self) -> dict:
        """Perform analysis on ID variables (CustomerID, ProductID, SellerID)."""
        results = {}
        
        for col in ID_VARS:
            freq = self._df[col].value_counts()
            income = self._df.groupby(col, observed=True)['TotalAmount'].sum()
            
            results[col] = {
                'frequency': freq,
                'income': income.reindex(freq.index),
                'top10_income': income.sort_values(ascending=False).head(10),
                'bottom10_income': income.sort_values(ascending=True).head(10),
            }
        
        return results
    
    def get_top_bottom_products_by_category(self) -> pd.DataFrame:
        """Get top and bottom products by income for each category."""
        # Merge to get ProductName
        df_full = self._df.merge(
            self._original_df[['ProductID', 'ProductName']].drop_duplicates(),
            on='ProductID',
            how='left'
        )
        
        plot_data = []
        
        for cat in df_full['Category'].unique():
            df_cat = df_full[df_full['Category'] == cat]
            
            prod_income = df_cat.groupby('ProductName')['TotalAmount'].sum()
            
            top1 = prod_income.sort_values(ascending=False).head(1)
            bottom1 = prod_income.sort_values(ascending=True).head(1)
            
            combined = pd.concat([top1, bottom1])
            
            for product, value in combined.items():
                plot_data.append({
                    'Category': cat,
                    'Product': product,
                    'TotalAmount': value,
                    'Type': 'Top 1' if product in top1.index else 'Bottom 1'
                })
        
        return pd.DataFrame(plot_data)
    
    def get_top_bottom_products_by_state(self) -> pd.DataFrame:
        """Get top and bottom products by income for each state."""
        df_full = self._df.merge(
            self._original_df[['ProductID', 'ProductName']].drop_duplicates(),
            on='ProductID',
            how='left'
        )
        
        plot_data = []
        
        for state in df_full['State'].unique():
            df_state = df_full[df_full['State'] == state]
            
            prod_income = df_state.groupby('ProductName')['TotalAmount'].sum()
            
            top1 = prod_income.sort_values(ascending=False).head(1)
            bottom1 = prod_income.sort_values(ascending=True).head(1)
            
            combined = pd.concat([top1, bottom1])
            
            for prod, income in combined.items():
                plot_data.append({
                    'State': state,
                    'Product': prod,
                    'TotalAmount': income,
                    'Type': 'Top 1' if prod in top1.index else 'Bottom 1'
                })
        
        return pd.DataFrame(plot_data)


def perform_eda(df: pd.DataFrame, original_df: Optional[pd.DataFrame] = None) -> ExploratoryDataAnalysis:
    """
    Convenience function to perform EDA.
    
    Parameters
    ----------
    df : pd.DataFrame
        The cleaned dataframe
    original_df : pd.DataFrame, optional
        The original dataframe
    
    Returns
    -------
    ExploratoryDataAnalysis
        The EDA instance
    """
    return ExploratoryDataAnalysis(df, original_df)
