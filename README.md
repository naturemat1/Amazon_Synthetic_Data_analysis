# Amazon Sales Data Analysis

A modular Python project for analyzing Amazon sales data with exploratory data analysis (EDA), visualizations, and customer segmentation using machine learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

## 📊 Project Overview

This project performs comprehensive analysis on Amazon sales data, including:

- **Data Loading & Cleaning**: Handle CSV data, remove duplicates, process missing values
- **Exploratory Data Analysis (EDA)**: Statistical summaries, distributions, correlations
- **Visualizations**: Histograms, boxplots, time series, scatter plots, pie charts, radar charts
- **Customer Segmentation**: K-Means clustering with PCA for customer profiling

## 🏗️ Architecture

The project follows a **Layered Architecture** pattern:

```
analisis-de-datos/
├── src/
│   ├── config/          # Configuration and settings
│   ├── core/           # Core utilities and formatters
│   ├── data/           # Data loading and cleaning
│   ├── analysis/       # EDA and statistics
│   ├── visualization/  # Plots and charts
│   ├── clustering/     # PCA, K-Means, segmentation
│   └── reports/        # Report generation
├── tests/              # Test files
├── notebooks/          # Jupyter notebooks
├── docs/               # Documentation
├── data/               # Data files (amazon.csv)
├── main.py             # Main entry point
├── requirements.txt    # Dependencies
├── LICENSE             # MIT License
└── README.md           # This file
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd analisis-de-datos
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Run Full Analysis
```bash
python main.py
```

#### Run Tests
```bash
pytest tests/
```

#### Use Jupyter Notebooks
```bash
jupyter notebook notebooks/analisis_exploratorio.ipynb
```

## 📦 Dependencies

- `pandas` - Data manipulation
- `matplotlib` - Plotting library
- `seaborn` - Statistical graphics
- `scikit-learn` - Machine learning
- `numpy` - Numerical computing

## 🎯 Features

### Data Analysis
- Basic statistics (numeric & categorical)
- Correlation matrix
- Top/Bottom analysis (products, customers, categories)
- Time series analysis

### Visualizations
- Distribution histograms and boxplots
- Categorical distribution with income overlay
- Monthly revenue time series
- Scatter plot matrices
- Correlation heatmaps
- Pie charts (cluster distribution)
- Radar charts (cluster profiles)
- PCA scatter plots with correlation circles

### Customer Segmentation
- Customer aggregation (RFM-like features)
- Standardization
- PCA for dimensionality reduction
- K-Means clustering
- Automatic segment labeling:
  - 🎯 PREMIUM (Alto Valor)
  - 🔄 FRECUENTES (Leales)
  - 💰 OCASIONALES (Sensibles Precio)
  - ⏰ INACTIVOS (Riesgo Pérdida)

## 🔧 Configuration

Edit `src/config/settings.py` to customize:

- Dataset path
- Column names and data types
- Variables for clustering
- K-Means and PCA parameters
- Plot styling

## 📁 Project Structure

| Directory | Description |
|-----------|-------------|
| `src/config/` | Configuration constants |
| `src/core/` | Utilities (formatters, display config) |
| `src/data/` | DataLoader and DataCleaner classes |
| `src/analysis/` | ExploratoryDataAnalysis and StatisticalAnalyzer |
| `src/visualization/` | DistributionPlotter and ChartPlotter |
| `src/clustering/` | PCAModel, KMeansModel, CustomerSegmenter |
| `src/reports/` | ReportGenerator for summaries |
| `tests/` | Unit tests |
| `notebooks/` | Jupyter notebooks |
| `docs/` | Project documentation |

## 📖 Documentation

- [Wiki](docs/wiki/Home.md) - Detailed documentation
- [Module Reference](docs/wiki/Home.md#modules-overview) - API documentation


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

Mateo Cobo 

## 🙏 Acknowledgments

- Original data source: Amazon sales dataset
- Built with: pandas, matplotlib, seaborn, scikit-learn

---

⭐ If you find this project useful, please give it a star!
