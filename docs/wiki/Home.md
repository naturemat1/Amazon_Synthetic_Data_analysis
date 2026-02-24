# Amazon Sales Data Analysis - Wiki

Welcome to the project wiki! This page contains detailed documentation about the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Modules Overview](#modules-overview)
4. [Configuration](#configuration)
5. [Running Tests](#running-tests)
6. [Contributing](#contributing)

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd analisis-de-datos
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

Run the full analysis:
```bash
python main.py
```

---

## Project Structure

```
analisis-de-datos/
├── src/
│   ├── config/          # Configuration settings
│   ├── core/           # Core utilities
│   ├── data/           # Data loading and cleaning
│   ├── analysis/       # EDA and statistics
│   ├── visualization/  # Plots and charts
│   ├── clustering/     # PCA, K-Means, segmentation
│   └── reports/        # Report generation
├── tests/              # Test files
├── notebooks/          # Jupyter notebooks
├── docs/               # Documentation
├── data/               # Data files
├── main.py             # Main entry point
└── requirements.txt    # Dependencies
```

---

## Modules Overview

### Data Layer (`src/data/`)

- **DataLoader**: Loads CSV data with proper data types
- **DataCleaner**: Handles duplicates, missing values, column removal

### Analysis Layer (`src/analysis/`)

- **ExploratoryDataAnalysis**: Statistical analysis and summaries
- **StatisticalAnalyzer**: Advanced statistics and cluster profiling

### Visualization Layer (`src/visualization/`)

- **DistributionPlotter**: Histograms, boxplots
- **ChartPlotter**: Time series, correlations, scatter plots

### Clustering Layer (`src/clustering/`)

- **PCAModel**: Principal Component Analysis
- **KMeansModel**: K-Means clustering
- **CustomerSegmenter**: Complete segmentation pipeline

---

## Configuration

Edit `src/config/settings.py` to customize:

- Dataset path
- Column names
- Clustering parameters
- Plot styling

---

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_data.py
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
