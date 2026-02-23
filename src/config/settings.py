"""
Configuration settings for the Amazon Sales Data Analysis project.
"""

# Dataset configuration
DATASET = "amazon.csv"

# Column names configuration
COLUMNS = {
    "id": {
        "order_id": "OrderID",
        "customer_id": "CustomerID",
        "customer_name": "CustomerName",
        "product_id": "ProductID",
        "product_name": "ProductName",
        "seller_id": "SellerID",
    },
    "categorical": {
        "category": "Category",
        "brand": "Brand",
        "payment_method": "PaymentMethod",
        "order_status": "OrderStatus",
        "city": "City",
        "state": "State",
        "country": "Country",
    },
    "numeric": {
        "quantity": "Quantity",
        "unit_price": "UnitPrice",
        "discount": "Discount",
        "tax": "Tax",
        "shipping_cost": "ShippingCost",
        "total_amount": "TotalAmount",
    },
    "date": "OrderDate",
}

# Data types for pandas
DTYPES = {
    "OrderID": "string",
    "CustomerID": "string",
    "CustomerName": "string",
    "ProductID": "string",
    "ProductName": "string",
    "Category": "category",
    "Brand": "category",
    "PaymentMethod": "category",
    "OrderStatus": "category",
    "City": "category",
    "State": "category",
    "Country": "category",
    "SellerID": "string",
}

# Columns to drop (irrelevant for analysis)
COLS_TO_DROP = [
    "OrderID",
    "CustomerName",
    "ProductName",
    "Country",
]

# Numeric variables for distribution analysis
NUMERIC_VARS = [
    "Quantity",
    "UnitPrice",
    "Discount",
    "Tax",
    "ShippingCost",
    "TotalAmount",
]

# Categorical variables for distribution analysis
CATEGORICAL_VARS = [
    "Category",
    "Brand",
    "PaymentMethod",
    "OrderStatus",
    "City",
    "State",
]

# ID variables for top/bottom analysis
ID_VARS = ["CustomerID", "ProductID", "SellerID"]

# Clustering variables
CLUSTERING_VARS = [
    "Ingreso_Total",
    "Ticket_Promedio",
    "Frecuencia",
    "Descuento_Promedio",
    "Dias_Ultima_Compra",
]

# K-Means configuration
KMEANS_CONFIG = {
    "n_clusters": 4,
    "init": "k-means++",
    "n_init": 50,
    "max_iter": 100,
    "algorithm": "lloyd",
    "random_state": 42,
}

# PCA configuration
PCA_CONFIG = {
    "n_components": 2,
    "random_state": 42,
}

# Visualization settings
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZES = {
    "small": (8, 5),
    "medium": (12, 6),
    "large": (16, 8),
    "extra_large": (20, 12),
}

# Output directory for plots
OUTPUT_DIR = "output"
