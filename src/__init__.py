"""
Amazon Sales Data Analysis - Modular Package

A modular data analysis project for Amazon sales data,
including EDA, visualizations, and customer segmentation.
"""

from . import config
from . import core
from . import data
from . import analysis
from . import visualization
from . import clustering
from . import reports

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "config",
    "core",
    "data",
    "analysis",
    "visualization",
    "clustering",
    "reports",
]
