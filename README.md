# Amazon Sales Data Analysis

A modular Python project for analyzing Amazon sales data with exploratory data analysis (EDA), visualizations, and customer segmentation using machine learning.

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
├── data/               # Data files (amazon.csv)
├── output/             # Generated visualizations
├── main.py             # Main entry point
└── requirements.txt    # Dependencies
```

### Module Descriptions

| Module | Description |
|--------|-------------|
| `config` | Configuration constants, column names, settings |
| `core` | Utilities (formatters, display config) |
| `data` | DataLoader and DataCleaner classes |
| `analysis` | ExploratoryDataAnalysis and StatisticalAnalyzer |
| `visualization` | DistributionPlotter and ChartPlotter |
| `clustering` | PCAModel, KMeansModel, CustomerSegmenter |
| `reports` | ReportGenerator for summaries |

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd analisis-de-datos
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate    # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 📦 Dependencies

```
pandas
matplotlib
seaborn
scikit-learn
numpy
```

## 🎯 Usage

### Run Full Analysis

```bash
python main.py
```

This will execute the complete pipeline:
1. Load and clean data
2. Perform EDA
3. Generate visualizations
4. Perform customer segmentation

### Run Specific Modules

You can also import and use individual modules:

```python
from src.data import DataLoader, DataCleaner
from src.visualization import DistributionPlotter
from src.clustering import CustomerSegmenter

# Load data
loader = DataLoader("amazon.csv")
df = loader.load()

# Clean data
cleaner = DataCleaner(df)
df_clean = cleaner.get_processed_data()

# Visualize distributions
plotter = DistributionPlotter(df_clean)
plotter.plot_numeric_distributions()

# Segment customers
segmenter = CustomerSegmenter(df_clean, original_df)
segmenter.perform_clustering(n_clusters=4)
```

## 📈 Features

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

## 📝 License

This project is for educational purposes.

## 👤 Author

Your Name

## 🙏 Acknowledgments

- Original data source: Amazon sales dataset
- Built with: pandas, matplotlib, seaborn, scikit-learn
