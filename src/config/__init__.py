"""Configuration package for Amazon Sales Data Analysis."""

from .settings import (
    DATASET,
    COLUMNS,
    DTYPES,
    COLS_TO_DROP,
    NUMERIC_VARS,
    CATEGORICAL_VARS,
    ID_VARS,
    CLUSTERING_VARS,
    KMEANS_CONFIG,
    PCA_CONFIG,
    PLOT_STYLE,
    FIGURE_SIZES,
    OUTPUT_DIR,
)

__all__ = [
    "DATASET",
    "COLUMNS",
    "DTYPES",
    "COLS_TO_DROP",
    "NUMERIC_VARS",
    "CATEGORICAL_VARS",
    "ID_VARS",
    "CLUSTERING_VARS",
    "KMEANS_CONFIG",
    "PCA_CONFIG",
    "PLOT_STYLE",
    "FIGURE_SIZES",
    "OUTPUT_DIR",
]
