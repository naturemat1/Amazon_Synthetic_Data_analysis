"""Analysis package for Amazon Sales Data Analysis.

This package handles exploratory data analysis and statistical operations.
"""

from .eda import ExploratoryDataAnalysis, perform_eda
from .statistics import StatisticalAnalyzer, analyze

__all__ = [
    "ExploratoryDataAnalysis",
    "perform_eda",
    "StatisticalAnalyzer",
    "analyze",
]
