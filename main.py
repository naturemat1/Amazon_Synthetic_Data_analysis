import pandas as pd     # Manejo y analisis de datos medinate tablas, leer archivos CSV, EXCEL.
import matplotlib.pyplot as plt # Crear graficos (Lineas, barras, dispersion, histogramas)
import numpy as np      # Manejo de arreglos y matrices, operaciones matematicas
import seaborn as sns   # Crear graficos al igual de plt
from sklearn.preprocessing import StandardScaler    # Permite escalar datos
from sklearn.decomposition import PCA               # Permite aplicar ACP
from sklearn.cluster import KMeans                  # Permite ejecutar el algoritmo de clusterizacion K-means
import itertools        # Libreria estandar para combinaciones y permutaciones (Para generar todas las combinaciones de los graficos de dispersión)
from matplotlib.ticker import FuncFormatter         # Permite personalziar el formato de los ejes en graficos
import matplotlib.dates as mdates                   # Permite manejar fechas en graficos, ejes temporales

DATASET = "amazon.csv"

# Formateo de valores numericos a miles (K) o millones (M)
def human_format(x, pos=None):
    if x >= 1_000_000:
        return f'{x/1_000_000:.1f}M'
    elif x >= 1_000:
        return f'{x/1_000:.1f}k'
    else:
        return f'{x:.0f}'

# =========================
# 1. CARGA DE DATOS
# =========================
DF = pd.read_csv(
    DATASET,
    parse_dates=["OrderDate"],
    dtype={
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
        "SellerID": "string"
    }
)

# --- CONFIGURACIÓN DE VISUALIZACIÓN EN PANDAS ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.3f}'.format)

# --- CONFIGURACIÓN DE ESTILO PARA GRÁFICOS ---
plt.style.use('seaborn-v0_8-darkgrid') # Define un estilo visual oscuro con cuadrícula para matplotlib

# --- INFORMACIÓN GENERAL DEL DATASET ---
print("\n" + "="*80)
print("1. DIMENSIONES ORIGINALES - DATASET:")
print("="*80)
print(f"   Shape: {DF.shape}")
print(f"   Total Registros: {DF.shape[0]:,}")
print(f"   Total Variables: {DF.shape[1]}")

print("\n1.1 Primeras y Ultimas 5 filas:")
print(DF.head())
print(DF.tail())

print(f"\n1.2 Tipos de Variables")
print(f"{DF.info()}\n")

# print(f"\n1.3 Valores Unicos por Variable")
# print(f"{DF.nunique()}\n")

# =========================
# 2. ESTADÍSTICAS BÁSICAS
# =========================
print("\n" + "="*80)
print("2.1 ESTADÍSTICAS BÁSICAS - Variables Numéricas:")
print("="*80)
print(DF.describe())

print("\n" + "="*80)
print("2.2 ESTADÍSTICAS BÁSICAS - Variables Categóricas:")
print("="*80)
print(DF.describe(include=["category", "string" ]).T)

# =========================
# 3. LIMPIEZA BÁSICA
# =========================
print("\n" + "="*80)
print("3. Eliminar registros duplicados:")
print("="*80)
duplicates = DF.duplicated().sum()
print(f"   Filas Duplicadas: {duplicates}")
if duplicates > 0:
    DF = DF.drop_duplicates() # Por defecto mantiene la primera ocurrencia y elimina las siguientes.
    print(f"   {duplicates} filas duplicadas eliminadas")
else:
    print("No se encontro filas duplicadas")
# =========================
# 4. VARIABLE FECHA - CONVERSIÓN
# =========================
print("\n" + "="*80)
print("4. Conversión de fecha de String a OrderDate:")
print("="*80)
# Conversión de fecha
DF['OrderDate'] = pd.to_datetime(DF['OrderDate'], errors='coerce')

# =========================
# 5. ELIMINACIÓN DE VARIABLES IRRELEVANTES
# =========================
drop_cols = [
    'OrderID',
    'CustomerName',
    'ProductName',
    'Country'
]
print("\n" + "="*80)
print("5. Eliminar variables irrelevantes")
print("="*80)
print(f"5.1 Variables eliminadas: {drop_cols}")
df = DF.drop(columns=drop_cols, errors='ignore')
# Ahora pasamos a usar df como variable para el analisis de Datos ya no DF

print("5.2 Variables restantes:", df.columns.tolist())

# =========================
# 6. TRATAMIENTO DE VALORES NULOS
# =========================
print("\n" + "="*80)
print(f"6. Revisión de Valores Nulos")
print("="*80)
missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({
    'Cantidad Nulos': missing_data,
    'Porcentaje': missing_percentage
})
missing_df = missing_summary[missing_summary['Cantidad Nulos'] > 0]
if len(missing_df) > 0:
    print(missing_df)
else:
    print(f"   6.1 Valores nulos: {len(missing_df)}")

# =========================
# 7. INGRESO TOTAL
# =========================
print("\n" + "="*80)
print(f"7. Ingreso Total")
print("="*80)

ingreso_total = DF["TotalAmount"].sum()
print(f"{ingreso_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# =========================
# 8. GRAFICAS DE DISTRIBUCION
# =========================
print("\n" + "="*80)
print(f"8. Graficas de Distribución")
print("="*80)

print(f"8.1 Graficas de Distribución - Variables Numéricas")
numeric_vars = [
    'Quantity',
    'UnitPrice',
    'Discount',
    'Tax',
    'ShippingCost',
    'TotalAmount'
]

for col in numeric_vars:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma
    sns.histplot(df[col], kde=True, ax=axes[0])
    axes[0].set_title(f'Distribución de {col}')

    # Boxplot con outliers rojos
    sns.boxplot(
        y=df[col],
        ax=axes[1],
        flierprops=dict(
            marker='o',
            markerfacecolor='red',
            markeredgecolor='red',
            markersize=5
        )
    )
    axes[1].set_title(f'Boxplot de {col}')

    plt.tight_layout()
    plt.show()

print(f"8.1 Graficas de Distribución - Variables Categóricas")
categorical_vars = [
    'Category',
    'Brand',
    'PaymentMethod',
    'OrderStatus',
    'City',
    'State'
]
for col in categorical_vars:

    freq = df[col].value_counts(sort=False)
    income = df.groupby(col, observed=True)['TotalAmount'].sum().reindex(freq.index)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # -----------------------------
    # Distribución (frecuencia)
    # -----------------------------
    sns.barplot(
        x=freq.index,
        y=freq.values,
        ax=ax1,
        color='#6BAED6'  # azul visible
    )
    ax1.set_ylabel('Frecuencia')
    ax1.set_xlabel(col)

    # Grid solo para frecuencia
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # -----------------------------
    # Ingresos (línea)
    # -----------------------------
    ax2 = ax1.twinx()
    ax2.plot(
        freq.index,
        income.values,
        color='#E31A1C',
        marker='o',
        linewidth=2
    )
    ax2.yaxis.set_major_formatter(FuncFormatter(human_format))
    ax2.set_ylabel('Ingresos')

    if col == 'Brand':
        income_sorted = income.sort_values(ascending=False)
        top5 = income_sorted.head(5).index
        bottom5 = income_sorted.tail(5).index
        show_labels = set(top5).union(bottom5)
    else:
        show_labels = freq.index

    # Etiquetas de ingresos
    for x, y in zip(freq.index, income.values):
        if x in show_labels:
            ax2.annotate(
                human_format(y, None),
                (x, y),
                textcoords='offset points',
                xytext=(0, 6),
                ha='center',
                fontsize=9,
                color='black'
            )

    ax1.set_title(f'Distribución e Ingresos por {col}')

    ax1.tick_params(
        axis='x',
        rotation=90 if col in ['Brand', 'Category', 'City', 'State'] else 45
    )

    plt.tight_layout()
    plt.show()

# =========================
# 9. GRAFICAS DE INGRESO X TIEMPO
# =========================
print("\n" + "="*80)
print(f"9. Gráficas de Ingreso x Tiempo")
print("="*80)

# Serie mensual
serie_mensual = df.set_index('OrderDate')['TotalAmount'].resample('ME').sum()

fig, ax = plt.subplots(figsize=(12, 5))

# Crear colores por año
years = serie_mensual.index.year
unique_years = sorted(years.unique())
colors_map = {year: color for year, color in zip(unique_years, plt.cm.tab10.colors)}

# Dibujar línea que conecta todos los puntos
ax.plot(serie_mensual.index, serie_mensual.values, color='gray', linewidth=1.5, alpha=0.6)

# Graficar cada punto con color según el año
for date, value in serie_mensual.items():
    ax.plot(date, value, marker='o', color=colors_map[date.year], markersize=6)

# Etiquetas encima de cada punto: inicial del mes
for date, value in serie_mensual.items():
    month_initial = date.strftime('%b')[0]  # primera letra del mes
    ax.text(date, value + serie_mensual.max()*0.01, month_initial,
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Configuración de eje X
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Año')
ax.set_ylabel('TotalAmount')
ax.set_title('TotalAmount mensual por año con iniciales de mes')
ax.yaxis.set_major_formatter(FuncFormatter(human_format))

# Grid solo horizontal
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()

# =========================
# 10. DISTRIBUCION DE INGRESOS X CUSTOMERID + TOP 20 y BOTTOM 20
# =========================
print("\n" + "="*80)
print(f"10. Distribución de Ingresos X CustomerID + TOP 20 y Bottom 20\n CustomerID', 'ProductID', 'SellerID'")
print("="*80)

id_vars = ['CustomerID', 'ProductID', 'SellerID']

for col in id_vars:
    # Frecuencia (distribución)
    freq = df[col].value_counts()
    order = freq.index
    
    # Ingresos
    income = (
        df.groupby(col)['TotalAmount']
          .sum()
          .reindex(order)
    )

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Barras: distribución
    sns.countplot(
        data=df,
        x=col,
        order=order,
        ax=ax1,
        color='#9ecae1'
    )
    ax1.set_ylabel('Frecuencia')
    ax1.set_xlabel('')
    ax1.set_xticks([])
    ax1.set_title(f'Distribución e Ingresos por {col}')

    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # Puntos rojos: ingresos - Top 20 y Bottom 20 ingresos
    income_sorted = income.sort_values(ascending=False)
    top20 = income_sorted.head(20).index
    bottom20 = income_sorted.tail(20).index

    selected_ids = [i for i in order if i in top20.union(bottom20)]
    selected_pos = [order.get_loc(i) for i in selected_ids]
    selected_income = income.loc[selected_ids]
    
    ax2 = ax1.twinx()
    ax2.scatter(
        selected_pos,
        selected_income.values,
        color='red',
        alpha=0.6,
        s=25,
        zorder=5
    )

    ax2.yaxis.set_major_formatter(FuncFormatter(human_format))
    ax2.set_ylabel('Ingresos')

    plt.tight_layout()
    plt.show()

# =========================
# 11. TOP 10 Y BOTTOM 10 de CustomerID', 'ProductID', 'SellerID'
# =========================
print("\n" + "="*80)
print(f"11. TOP 10 Y BOTTOM 10\nCustomerID', 'ProductID', 'SellerID'")
print("="*80)
for col in id_vars:
    # Sumar ingresos por ID
    income = df.groupby(col)['TotalAmount'].sum().sort_values(ascending=False)
    top10 = income.head(10)
    bottom10 = income.tail(10)
    combined = pd.concat([top10, bottom10])

    # Reemplazar IDs por nombres
    labels = pd.Series(combined.index.astype(str))  # convertir a Series para map

    if col == 'CustomerID':
        name_map = DF[['CustomerID', 'CustomerName']].drop_duplicates().set_index('CustomerID')['CustomerName']
        labels = labels.map(lambda x: name_map.get(x, x))

    elif col == 'ProductID':
        name_map = DF[['ProductID', 'ProductName']].drop_duplicates().set_index('ProductID')['ProductName']
        labels = labels.map(lambda x: name_map.get(x, x))

    # SellerID no tiene nombre → se deja tal cual

    # Gráfica de barras
    plt.figure(figsize=(14, 5))
    ax = sns.barplot(
        x=labels,
        y=combined.values,
        palette=['red'] * 10 + ['blue'] * 10
    )

    plt.title(f'Top 10 y Bottom 10 Ingresos por {col}')
    plt.xlabel('')
    ax.yaxis.set_major_formatter(FuncFormatter(human_format))
    plt.ylabel('Ingresos')
    plt.xticks(rotation=90)

    # Etiquetas encima de las barras
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{height:,.1f}',
            (p.get_x() + p.get_width() / 2., height),
            ha='center',
            va='bottom',
            fontsize=9,
            xytext=(0, 3),
            textcoords='offset points'
        )

    plt.tight_layout()
    plt.show()

# =========================
# 12. Top 1 y Bottom 1 de Productos x Ingresos x Categoría
# =========================
# Unir df con DF para obtener ProductName
df_full = df.merge(
    DF[['ProductID', 'ProductName']].drop_duplicates(),
    on='ProductID',
    how='left'
)

print("\n" + "="*80)
print(f"12. TOP 1 Y BOTTOM 1 de Productos x Ingresos x Categoría'")
print("="*80)

# Lista para almacenar los datos de todas las categorías
plot_data = []

# Iterar por categorías
for cat in df_full['Category'].unique():
    df_cat = df_full[df_full['Category'] == cat]
    
    # Sumar ingresos por producto
    prod_income = df_cat.groupby('ProductName')['TotalAmount'].sum()
    
    # Top 1 y Bottom 1
    top1 = prod_income.sort_values(ascending=False).head(1)
    bottom1 = prod_income.sort_values(ascending=True).head(1)
    
    combined = pd.concat([top1, bottom1])
    
    # Guardar los datos en formato de lista de diccionarios
    for product, value in combined.items():
        plot_data.append({
            'Category': cat,
            'Product': product,
            'TotalAmount': value,
            'Type': 'Top 1' if product in top1.index else 'Bottom 1'
        })

# Convertir en DataFrame
df_plot = pd.DataFrame(plot_data)

# Crear etiquetas combinando categoría y producto
df_plot['Label'] = df_plot['Category'] + ' - ' + df_plot['Product']

# Colores
palette = {'Top 1': 'red', 'Bottom 1': 'blue'}

# Gráfico único
plt.figure(figsize=(16,6))
ax = sns.barplot(
    x='Label',
    y='TotalAmount',
    hue='Type',
    data=df_plot,
    palette=palette
)

# Formato eje y
ax.yaxis.set_major_formatter(FuncFormatter(human_format))
plt.xticks(rotation=90, ha='right')
plt.ylabel('Ingresos')
plt.xlabel('Producto - Categoría')
plt.title('Top 1 y Bottom 1 productos por categoría')

# Etiquetas encima de cada barra
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        human_format(height, None),
        (p.get_x() + p.get_width()/2, height),
        ha='center',
        va='bottom',
        fontsize=9,
        xytext=(0,3),
        textcoords='offset points'
    )

plt.tight_layout()
plt.show()

# =========================
# 13. Top 1 y Bottom 1 de Productos x Ingresos x Estado
# =========================
print("\n" + "="*80)
print(f"13. TOP 1 Y BOTTOM 1 de Productos x Ingresos x Estado'")
print("="*80)

# Lista para guardar datos de todos los estados
plot_data = []

# Iteramos por cada estado
for state in df_full['State'].unique():
    df_state = df_full[df_full['State'] == state]
    
    # Sumar ingresos totales por producto
    prod_income = df_state.groupby('ProductName')['TotalAmount'].sum()
    
    # Top 1 y Bottom 1
    top1 = prod_income.sort_values(ascending=False).head(1)
    bottom1 = prod_income.sort_values(ascending=True).head(1)
    
    combined = pd.concat([top1, bottom1])
    
    # Guardar en lista
    for prod, income in combined.items():
        plot_data.append({
            'State': state,
            'Product': prod,
            'TotalAmount': income,
            'Type': 'Top 1' if prod in top1.index else 'Bottom 1'
        })

# Convertir a DataFrame
df_plot = pd.DataFrame(plot_data)

# Crear etiquetas combinando producto y estado
df_plot['Label'] = df_plot['Product'] + '\n(' + df_plot['State'] + ')'

# Colores
palette = {'Top 1': 'red', 'Bottom 1': 'blue'}

# Gráfico vertical
plt.figure(figsize=(18,6))
ax = sns.barplot(
    x='Label',
    y='TotalAmount',
    hue='Type',
    data=df_plot,
    palette=palette
)

plt.title('Top 1 y Bottom 1 productos por estado')
plt.xlabel('Producto (Estado)')
plt.ylabel('Ingresos')
ax.yaxis.set_major_formatter(FuncFormatter(human_format))

# Etiquetas encima de cada barra
for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        human_format(height, None),
        (p.get_x() + p.get_width()/2, height),
        ha='center',
        va='bottom',
        fontsize=9,
        xytext=(0,3),
        textcoords='offset points'
    )

plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.show()

# =========================
# 14. PRODUCTOS MAS VENDIDOS
# =========================
print("\n" + "="*80)
print(f"14. TOP 5 Productos más vendidos")
print("="*80)

resumen_productos_mas_vendidos = (
    DF.groupby("ProductName")
      .agg(
          UnidadesVendidas=("Quantity", "sum"),
          IngresoTotal=("TotalAmount", "sum")
      )
      .sort_values("UnidadesVendidas", ascending=False)
      .head(5)
)
print(resumen_productos_mas_vendidos)

# =========================
# 15. PRODUCTOS CON MAS INGRESOS
# =========================
print("\n" + "="*80)
print(f"15. TOP 5 Productos que generaron más ingresos")
print("="*80)

top_productos_ingresos = (
    DF.groupby("ProductName")
      .agg(
          UnidadesVendidas=("Quantity", "sum"),
          IngresoTotal=("TotalAmount", "sum")
      )
      .sort_values("IngresoTotal", ascending=False)
      .head(5)
)
print(top_productos_ingresos)

# =========================
# 16. CLIENTES CON MAS INGRESOS
# =========================
print("\n" + "="*80)
print(f"16. TOP 5 Clientes que generaron más ingresos")
print("="*80)
top_clientes_ingresos = (
    DF.groupby(["CustomerID", "CustomerName"])
      .agg(
          TotalCompras=("Quantity", "sum"),
          IngresoTotal=("TotalAmount", "sum")
      )
      .sort_values("IngresoTotal", ascending=False)
      .head(5)
)
print(top_clientes_ingresos)

# =========================
# 17. MATRIZ DE CORRELACIONES
# =========================
print("\n" + "="*80)
print("17. MATRIZ DE CORRELACIONES (Variables Numéricas)")
print("="*80)

# Seleccionar solo columnas numéricas
df_numeric = DF.select_dtypes(include=['number'])
# Matriz de correlación
corr_matrix = df_numeric.corr()
print(corr_matrix.round(3))

# HEATMAP BICOLOR
fig, ax = plt.subplots(figsize=(10, 8))

# Colormap bicolor
cmap = plt.cm.RdBu  # Rojo ↔ Blanco ↔ Azul

im = ax.imshow(
    corr_matrix,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    interpolation='nearest'
)

# Barra de color
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlación')

# Ticks
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_matrix.columns)

# Valores dentro de las celdas
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        ax.text(
            j, i,
            f"{corr_matrix.iloc[i, j]:.2f}",
            ha='center',
            va='center',
            color='black',
            fontsize=9
        )

ax.set_title("Matriz de Correlaciones (Bicolor)")
plt.tight_layout()
plt.show()

# =========================
# 18. GRAFICAS DE DISPERSION
# =========================
print("\n" + "="*80)
print(f"18. Graficas de Dispersión")
print("="*80)

pairplot_cols = ['Quantity', 'UnitPrice', 'Discount', 'Tax', 'ShippingCost', 'TotalAmount']
if set(pairplot_cols).issubset(df.columns):
    data = df[pairplot_cols].dropna()

    # Todas las combinaciones sin repetición ni diagonal
    pairs = list(itertools.combinations(pairplot_cols, 2))

    # Crear figura 3x5
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for ax, (y, x) in zip(axes, pairs):
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            ax=ax
        )
        ax.set_xlabel(x)
        ax.set_ylabel(y)

    # Eliminar ejes vacíos (por seguridad)
    for i in range(len(pairs), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(
        'Diagramas de Dispersión Bivariados (3 filas × 5 columnas)',
        fontsize=16
    )

    fig.tight_layout()
    fig.savefig(
        "Diagrama_dispersion_3x5_bivariado.png",
        dpi=300,
        bbox_inches="tight"
    )
else:
    print('Faltan algunas de las columnas requeridas para el gráfico de pares.')

# =========================
# 19. AGREGACIÓN POR CLIENTE
# =========================
print("\n" + "="*80)
print(f"19. Seleccion de variables para clusterización")
print("="*80)

clientes_agg = df.groupby('CustomerID').agg({
    'TotalAmount': ['sum', 'mean', 'count'],
    'Quantity': 'sum',
    'Discount': 'mean',
    'UnitPrice': 'mean',
    'OrderDate': 'max'
}).round(2)

clientes_agg.columns = [
    'Ingreso_Total', 'Ticket_Promedio', 'Frecuencia',
    'Cantidad_Total', 'Descuento_Promedio',
    'Precio_Promedio', 'Ultima_Compra'
]

clientes_agg['Dias_Ultima_Compra'] = (
    pd.Timestamp.now() - clientes_agg['Ultima_Compra']
).dt.days

print(f"Clientes analizados: {len(clientes_agg)}")

clustering_vars = [
    'Ingreso_Total',
    'Ticket_Promedio',
    'Frecuencia',
    'Descuento_Promedio',
    'Dias_Ultima_Compra'
]

# =========================
# 20. ESTANDARIZACIÓN
# =========================
print("\n" + "="*80)
print("20. ESTANDARIZACIÓN - Centrar y Reducir")
print("="*80)

scaler = StandardScaler()
clientes_scaled = pd.DataFrame(
    scaler.fit_transform(clientes_agg[clustering_vars].fillna(0)),
    columns=clustering_vars,
    index=clientes_agg.index
)

print(clientes_scaled.describe().loc[['mean', 'std']])

# =========================
# 21. ACP
# =========================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(clientes_scaled)

explained_var = pca.explained_variance_ratio_

print("\n" + "="*80)
print("21. ACP")
print("="*80)
print(f"\nVarianza explicada PC1 + PC2: {explained_var.sum():.2%}")

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=clustering_vars
)

print(loadings)

print("21.1 Grafica ACP de Clientes (Sin Clustering)")

plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6, color='gray')

plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
plt.title("Proyección ACP de Clientes (Sin Clustering)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================
# 23. MÉTODO DEL CODO (INERCIA)
# =========================
print("\n" + "="*80)
print("23. Método Codo Jambú")
print("="*80)

K_range = range(2, 11)
inertia = []

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=50,
        max_iter=100,
        algorithm='lloyd',
        random_state=42
    )
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del Codo (Clientes - ACP)")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 24. K-MEANS
# =========================
print("\n" + "="*80)
print("24. K-MEANS")
print("="*80)

k_opt = 4

kmeans = KMeans(
    n_clusters=k_opt,
    init='k-means++',
    n_init=50,
    max_iter=100,
    algorithm='lloyd',
    random_state=42
)

clientes_agg['Segmento'] = kmeans.fit_predict(X_pca)

# Inercia
inertia_intra = kmeans.inertia_
global_center = X_pca.mean(axis=0)
inertia_total = np.sum((X_pca - global_center) ** 2)
inertia_inter = inertia_total - inertia_intra

print("24.1. ANÁLISIS DE INERCIAS")
print(f"Inercia total:      {inertia_total:.4f}")
print(f"Inercia intraclase: {inertia_intra:.4f} ({100*inertia_intra/inertia_total:.2f}%)")
print(f"Inercia interclase: {inertia_inter:.4f} ({100*inertia_inter/inertia_total:.2f}%)")

# =========================
# 25. PCA + CLUSTERS + CÍRCULO DE CORRELACIONES (ETIQUETAS AUTOMÁTICAS + RADIO = 10)
# =========================
plt.rcParams['font.family'] = 'Segoe UI Emoji'

print("25. PCA + CLUSTERS + CÍRCULO DE CORRELACIONES (ETIQUETAS AUTOMÁTICAS + RADIO = 10)")
# ----------------------------
# 8.1 PERFIL DE CLUSTERS
# ----------------------------
perfil_segmentos = clientes_agg.groupby('Segmento')[clustering_vars].mean()
# Relativo al promedio global
perfil_relativo = perfil_segmentos / clientes_agg[clustering_vars].mean()
# ----------------------------
# ASIGNACIÓN FLEXIBLE DE ETIQUETAS
# ----------------------------
segmento_labels = {}
for seg, row in perfil_relativo.iterrows():

    # 1. Premium: alto ingreso + alta frecuencia + reciente
    if row['Ingreso_Total'] > 1.2 and row['Frecuencia'] > 1.2 and row['Dias_Ultima_Compra'] < 0.8:
        segmento_labels[seg] = '🎯 PREMIUM (Alto Valor)'

    # 2. Frecuentes: alta frecuencia (aunque ingreso medio)
    elif row['Frecuencia'] > 1.1:
        segmento_labels[seg] = '🔄 FRECUENTES (Leales)'

    # 3. Ocasionales / Sensibles al precio: alta sensibilidad a descuento
    elif row['Descuento_Promedio'] > 1.05:
        segmento_labels[seg] = '💰 OCASIONALES (Sensibles Precio)'

    # 4. Inactivos: recencia alta + frecuencia baja
    else:
        segmento_labels[seg] = '⏰ INACTIVOS (Riesgo Pérdida)'

print("\nEtiquetas asignadas automáticamente (flexible):")
for k, v in segmento_labels.items():
    print(f"Cluster {k}: {v}")


# PLOTEO FINAL
centroids = kmeans.cluster_centers_
fig, ax = plt.subplots(figsize=(9, 8))

# PCA + CLUSTERS
for c in range(k_opt):
    ax.scatter(
        X_pca[clientes_agg['Segmento'] == c, 0],
        X_pca[clientes_agg['Segmento'] == c, 1],
        label=segmento_labels[c],
        alpha=0.6
    )

# Centroides
ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker='X',
    s=200,
    color='black',
    label='Centroides'
)

ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
ax.set_title("Segmentación de Clientes (PCA + Círculo de Correlaciones)")
ax.grid(alpha=0.3)

# Guardar límites reales del PCA
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 2. CÍRCULO DE CORRELACIONES CENTRADO EN (0,0)
# Escala visual del círculo (NO afecta datos)
circle_scale = min(
    (xlim[1] - xlim[0]),
    (ylim[1] - ylim[0])
) * 0.9   # ajusta tamaño aquí

theta = np.linspace(0, 2*np.pi, 300)

# Círculo
ax.plot(
    circle_scale * np.cos(theta),
    circle_scale * np.sin(theta),
    linestyle='--',
    color='gray',
    alpha=0.6
)

# Vectores de carga
for var in loadings.index:
    x = loadings.loc[var, 'PC1'] * circle_scale
    y = loadings.loc[var, 'PC2'] * circle_scale

    ax.arrow(
        0, 0, x, y,
        color='black',
        width=0.0,
        head_width=circle_scale * 0.06,
        length_includes_head=True,
        alpha=0.85
    )

    ax.text(
        x * 1.1,
        y * 1.1,
        var,
        fontsize=9,
        ha='center',
        va='center'
    )

# Ejes base
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# Forzar rango del PCA
ax.set_xlim(-10, 30)
ax.set_ylim(-10, 10)

ax.legend()
plt.tight_layout()
plt.show()

# ============================
# Gráfico de pastel: Distribución de registros por cluster
# ============================

# Contar registros en df_full para cada cluster
record_counts = []

for c in range(k_opt):
    cluster_customers = clientes_agg[clientes_agg['Segmento'] == c].index
    n_records = df_full[df_full['CustomerID'].isin(cluster_customers)].shape[0]
    record_counts.append(n_records)

record_counts = np.array(record_counts)
record_labels = [f"Cluster {i}\n{segmento_labels[i]}" for i in range(k_opt)]

# Graficar
plt.figure(figsize=(8,8))
plt.pie(
    record_counts,
    labels=record_labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.tab10.colors[:k_opt],
    wedgeprops={'edgecolor': 'black'}
)
plt.title("Distribución de registros por segmento (K-Means)")
plt.show()

# ============================
# Gráfico de pastel: Distribución de segmentos
# ============================

segment_counts = clientes_agg['Segmento'].value_counts().sort_index()
segment_labels = [f"Cluster {i}\n{segmento_labels[i]}" for i in segment_counts.index]

plt.figure(figsize=(8,8))
plt.pie(
    segment_counts,
    labels=segment_labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=plt.cm.tab10.colors[:len(segment_counts)],
    wedgeprops={'edgecolor': 'black'}
)
plt.title("Distribución de clientes por segmentos (K-Means)")
plt.show()

# ============================
# Gráfico de radar
# ============================

clustering_vars = [
    'Ingreso_Total',
    'Ticket_Promedio',
    'Frecuencia',
    'Descuento_Promedio',
    'Dias_Ultima_Compra'
]

# Promedios por cluster
cluster_means = clientes_agg.groupby('Segmento')[clustering_vars].mean()

# Normalizar cada variable para que estén entre 0 y 1
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

# Preparar ángulos
num_vars = len(clustering_vars)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # cerrar el círculo

plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

# Dibujar círculos concéntricos
ax.set_rscale('linear')
ax.set_ylim(0, 1.1)

# Variables como etiquetas
plt.xticks(angles[:-1], clustering_vars, fontsize=10)

# Graficar cada cluster
colors = plt.cm.tab10.colors
for i, row in cluster_means_norm.iterrows():
    values = row.tolist()
    values += values[:1]  # cerrar el círculo
    ax.plot(angles, values, color=colors[i], linewidth=2, label=f"Cluster {i} ({segmento_labels[i]})")
    ax.fill(angles, values, color=colors[i], alpha=0.25)  # relleno semitransparente

# Círculos de referencia
ax.yaxis.grid(True, color='gray', linestyle='--', alpha=0.5)
ax.xaxis.grid(True, color='gray', linestyle='--', alpha=0.5)

plt.title("Perfil de clusters (Radar / Estrella)", size=14, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.show()

# ==============================
# Perfil detallado por cluster (extendido)
# ==============================
cluster_stats = []

for c in range(k_opt):
    # Tomamos los clientes de este cluster
    cluster_customers = clientes_agg[clientes_agg['Segmento'] == c].index
    
    # Filtrar registros de df_full usando CustomerID
    df_cluster = df_full[df_full['CustomerID'].isin(cluster_customers)]
    
    n_records = len(df_cluster)
    pct_total = n_records / len(df_full) * 100
    total_ingresos = df_cluster['TotalAmount'].sum()
    
    # Rango de precios
    min_price = df_cluster['UnitPrice'].min()
    max_price = df_cluster['UnitPrice'].max()
    
    # Top 3 y Bottom 3 productos por ingreso
    prod_income = df_cluster.groupby('ProductName')['TotalAmount'].sum()
    top3_prod = prod_income.sort_values(ascending=False).head(3).to_dict()
    bottom3_prod = prod_income.sort_values(ascending=True).head(3).to_dict()
    
    # Top 3 y Bottom 3 categorías por ingreso
    cat_income = df_cluster.groupby('Category')['TotalAmount'].sum()
    top3_cat = cat_income.sort_values(ascending=False).head(3).to_dict()
    bottom3_cat = cat_income.sort_values(ascending=True).head(3).to_dict()
    
    # Top 3 y Bottom 3 productos por cantidad
    prod_qty = df_cluster.groupby('ProductName')['Quantity'].sum()
    top3_qty_prod = prod_qty.sort_values(ascending=False).head(3).to_dict()
    bottom3_qty_prod = prod_qty.sort_values(ascending=True).head(3).to_dict()
    
    # Top 3 y Bottom 3 categorías por cantidad
    cat_qty = df_cluster.groupby('Category')['Quantity'].sum()
    top3_qty_cat = cat_qty.sort_values(ascending=False).head(3).to_dict()
    bottom3_qty_cat = cat_qty.sort_values(ascending=True).head(3).to_dict()
    
    # Estados presentes
    estados = df_cluster['State'].value_counts().to_dict()
    
    # Estadísticas por usuario
    user_stats = df_cluster.groupby('CustomerID').agg({'Quantity':'sum', 'TotalAmount':'sum'})
    
    # Usuario con más compras (cantidad)
    top_user_qty = user_stats['Quantity'].idxmax()
    top_user_qty_value = user_stats.loc[top_user_qty, 'Quantity']
    top_user_qty_income = user_stats.loc[top_user_qty, 'TotalAmount']
    
    # Usuario con menos compras (cantidad)
    bottom_user_qty = user_stats['Quantity'].idxmin()
    bottom_user_qty_value = user_stats.loc[bottom_user_qty, 'Quantity']
    bottom_user_qty_income = user_stats.loc[bottom_user_qty, 'TotalAmount']
    
    # Usuario con más ingreso
    top_user_income = user_stats['TotalAmount'].idxmax()
    top_user_income_value = user_stats.loc[top_user_income, 'TotalAmount']
    top_user_income_qty = user_stats.loc[top_user_income, 'Quantity']
    
    # Usuario con menos ingreso
    bottom_user_income = user_stats['TotalAmount'].idxmin()
    bottom_user_income_value = user_stats.loc[bottom_user_income, 'TotalAmount']
    bottom_user_income_qty = user_stats.loc[bottom_user_income, 'Quantity']
    
    # Cantidad de clientes únicos
    n_customers = df_cluster['CustomerID'].nunique()
    
    # Cantidad de productos únicos
    n_products = df_cluster['ProductID'].nunique() if 'ProductID' in df_cluster.columns else df_cluster['ProductName'].nunique()
    
    # Cantidad de vendedores únicos
    n_sellers = df_cluster['SellerID'].nunique() if 'SellerID' in df_cluster.columns else None
    
    # Guardar todo en diccionario
    cluster_stats.append({
        'Cluster': c,
        'Etiqueta': segmento_labels[c],
        'Registros': n_records,
        'Porcentaje_total': pct_total,
        'Clientes_unicos': n_customers,
        'Productos_unicos': n_products,
        'Vendedores_unicos': n_sellers,
        'Ingresos_totales': total_ingresos,
        'Precio_min': min_price,
        'Precio_max': max_price,
        'Top3_Productos_Ingreso': top3_prod,
        'Bottom3_Productos_Ingreso': bottom3_prod,
        'Top3_Categorias_Ingreso': top3_cat,
        'Bottom3_Categorias_Ingreso': bottom3_cat,
        'Top3_Productos_Cantidad': top3_qty_prod,
        'Bottom3_Productos_Cantidad': bottom3_qty_prod,
        'Top3_Categorias_Cantidad': top3_qty_cat,
        'Bottom3_Categorias_Cantidad': bottom3_qty_cat,
        'Estados': estados,
        'Usuario_Mas_Compras': {'CustomerID': top_user_qty, 'Cantidad': top_user_qty_value, 'Ingresos': top_user_qty_income},
        'Usuario_Menos_Compras': {'CustomerID': bottom_user_qty, 'Cantidad': bottom_user_qty_value, 'Ingresos': bottom_user_qty_income},
        'Usuario_Mas_Ingreso': {'CustomerID': top_user_income, 'Ingresos': top_user_income_value, 'Cantidad': top_user_income_qty},
        'Usuario_Menos_Ingreso': {'CustomerID': bottom_user_income, 'Ingresos': bottom_user_income_value, 'Cantidad': bottom_user_income_qty},
    })

# Convertir a DataFrame para mejor visualización
df_clusters = pd.DataFrame(cluster_stats)

# Mostrar el resumen completo
pd.set_option('display.max_colwidth', None)
print(df_clusters)

# Función para formatear números grandes
def human_format(num, pos=None):
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000:
            return f"{num:.0f}{unit}"
        num /= 1000
    return f"{num:.0f}B"

# Generar gráficos por cluster
for c in range(k_opt):
    df_cluster = df_full[df_full['CustomerID'].isin(clientes_agg[clientes_agg['Segmento']==c].index)]
    etiqueta = segmento_labels[c]
    
    # --- Productos por ingreso ---
    prod_income = df_cluster.groupby('ProductName')['TotalAmount'].sum()
    top3 = prod_income.sort_values(ascending=False).head(3)
    bottom3 = prod_income.sort_values(ascending=True).head(3)
    combined = pd.concat([top3, bottom3])
    
    labels = combined.index
    colors = ['red']*3 + ['blue']*3
    
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=labels, y=combined.values, palette=colors)
    plt.title(f'Cluster {c} ({etiqueta}) - Top/Bottom 3 Productos por Ingreso')
    plt.ylabel('Total Amount')
    plt.xlabel('Producto')
    
    # Etiquetas encima de las barras
    for p in ax.patches:
        ax.annotate(human_format(p.get_height()),
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # --- Productos por cantidad ---
    prod_qty = df_cluster.groupby('ProductName')['Quantity'].sum()
    top3_qty = prod_qty.sort_values(ascending=False).head(3)
    bottom3_qty = prod_qty.sort_values(ascending=True).head(3)
    combined_qty = pd.concat([top3_qty, bottom3_qty])
    
    labels = combined_qty.index
    colors = ['red']*3 + ['blue']*3
    
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=labels, y=combined_qty.values, palette=colors)
    plt.title(f'Cluster {c} ({etiqueta}) - Top/Bottom 3 Productos por Cantidad')
    plt.ylabel('Cantidad')
    plt.xlabel('Producto')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # --- Categorías por ingreso ---
    cat_income = df_cluster.groupby('Category')['TotalAmount'].sum()
    top3_cat = cat_income.sort_values(ascending=False).head(3)
    bottom3_cat = cat_income.sort_values(ascending=True).head(3)
    combined_cat = pd.concat([top3_cat, bottom3_cat])
    
    labels = combined_cat.index
    colors = ['red']*3 + ['blue']*3
    
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=labels, y=combined_cat.values, palette=colors)
    plt.title(f'Cluster {c} ({etiqueta}) - Top/Bottom 3 Categorías por Ingreso')
    plt.ylabel('Total Amount')
    plt.xlabel('Categoría')
    
    for p in ax.patches:
        ax.annotate(human_format(p.get_height()),
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # --- Categorías por cantidad ---
    cat_qty = df_cluster.groupby('Category')['Quantity'].sum()
    top3_cat_qty = cat_qty.sort_values(ascending=False).head(3)
    bottom3_cat_qty = cat_qty.sort_values(ascending=True).head(3)
    combined_cat_qty = pd.concat([top3_cat_qty, bottom3_cat_qty])
    
    labels = combined_cat_qty.index
    colors = ['red']*3 + ['blue']*3
    
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=labels, y=combined_cat_qty.values, palette=colors)
    plt.title(f'Cluster {c} ({etiqueta}) - Top/Bottom 3 Categorías por Cantidad')
    plt.ylabel('Cantidad')
    plt.xlabel('Categoría')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()