import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🔍 Iniciando Análisis Exploratorio - AI Tool Adoption")

# Cargar dataset - VERIFICAR RUTA
try:
    df = pd.read_csv('data/ai_adoption_dataset.csv')
    print("✅ Dataset cargado correctamente")
except FileNotFoundError:
    # Intentar rutas alternativas
    try:
        df = pd.read_csv('../data/ai_adoption_dataset.csv')
        print("✅ Dataset cargado correctamente (desde ../data/)")
    except:
        print("❌ No se pudo cargar el dataset. Verifica la ruta.")
        exit()

# Inspección inicial
print("📊 DIMENSIONES DEL DATASET:")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

print("\n📋 PRIMERAS FILAS:")
print(df.head())

print("\n🔍 INFORMACIÓN GENERAL:")
print(df.info())

print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
print(df.describe(include='all'))

print("\n🎯 ESTRUCTURA DE COLUMNAS:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col} - {df[col].dtype}")
   
    
# Función para análisis de valores nulos
def analizar_valores_nulos(df):
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df)) * 100
    
    print("📊 ANÁLISIS DE VALORES NULOS:")
    for col in df.columns:
        print(f"{col}: {nulos[col]} nulos ({porcentaje_nulos[col]:.2f}%)")
    
    return nulos

# Analizar valores nulos
nulos = analizar_valores_nulos(df)

# Limpieza básica
df_clean = df.copy()

# Eliminar columnas con más del 50% de valores nulos
umbral_nulos = 0.5
columnas_a_eliminar = nulos[nulos > len(df) * umbral_nulos].index
df_clean = df_clean.drop(columns=columnas_a_eliminar)

print(f"🗑️ Columnas eliminadas: {list(columnas_a_eliminar)}")

# Llenar valores nulos según el tipo de dato
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype in ['float64', 'int64']:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
        else:
            df_clean[col].fillna('Desconocido', inplace=True)

print("✅ Limpieza de datos completada")

def analisis_univariado(df):
    print("📊 ANÁLISIS UNIVARIADO")
    
    # Seleccionar columnas numéricas y categóricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"🔢 Columnas numéricas: {len(numeric_cols)}")
    print(f"📝 Columnas categóricas: {len(categorical_cols)}")
    
    return numeric_cols, categorical_cols

numeric_cols, categorical_cols = analisis_univariado(df_clean)

# CORRECCIÓN 1: Visualización de distribuciones numéricas MEJORADA
if len(numeric_cols) > 0:
    print("\n📊 CREANDO GRÁFICOS DE DISTRIBUCIÓN...")
    
    # Crear figura con subplots adecuados
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    
    # Aplanar el array de axes para iteración fácil
    if n_rows > 1 or n_cols > 1:
        axes = axes.ravel()
    else:
        axes = [axes]
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            # Tratamiento especial para 'year' - usar gráfico de barras
            if col == 'year':
                df_clean[col].value_counts().sort_index().plot(kind='bar', ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribución de {col} (Variable Categórica)')
            else:
                df_clean[col].hist(bins=30, ax=axes[i], alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribución de {col}')
            
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frecuencia')
    
    # Ocultar ejes vacíos
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('distribuciones_numericas.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico de distribuciones guardado: distribuciones_numericas.png")
    plt.show()

# Análisis de variables categóricas
if len(categorical_cols) > 0:
    print("\n📝 ANÁLISIS DE VARIABLES CATEGÓRICAS:")
    for col in categorical_cols:
        # Solo analizar columnas con menos de 50 valores únicos
        if df_clean[col].nunique() <= 50:
            print(f"\n📊 Análisis de {col}:")
            print(f"Valores únicos: {df_clean[col].nunique()}")
            print(f"Top 5 valores:")
            print(df_clean[col].value_counts().head())

# CORRECCIÓN 2: Análisis de correlación MEJORADO
if len(numeric_cols) > 1:
    print("\n🔥 CREANDO MAPA DE CORRELACIONES...")
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_clean[numeric_cols].corr()
    
    # Crear máscara para el triángulo superior
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.3f',  # 3 decimales para ver mejor
                cbar_kws={"shrink": .8})
    plt.title('Mapa de Correlación de Variables Numéricas')
    plt.tight_layout()
    plt.savefig('correlaciones.png', dpi=300, bbox_inches='tight')
    print("✅ Mapa de correlaciones guardado: correlaciones.png")
    plt.show()
    
    # Identificar correlaciones fuertes
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                strong_correlations.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
    
    print("\n🔥 CORRELACIONES FUERTES (>0.7):")
    if strong_correlations:
        for corr in strong_correlations:
            print(f"  {corr[0]} - {corr[1]}: {corr[2]:.3f}")
    else:
        print("  No se encontraron correlaciones fuertes")

# CORRECCIÓN 3: Análisis por industria (CON NOMBRES CORRECTOS)
if 'industry' in df_clean.columns and 'adoption_rate' in df_clean.columns:
    print("\n🏭 CREANDO ANÁLISIS POR INDUSTRIA...")
    
    plt.figure(figsize=(12, 8))
    industry_adoption = df_clean.groupby('industry')['adoption_rate'].mean().sort_values(ascending=False)
    
    industry_adoption.plot(kind='bar')
    plt.title('Tasa de Adopción de IA por Industria')
    plt.xlabel('Industria')
    plt.ylabel('Tasa de Adopción Promedio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('adopcion_por_industria.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico por industria guardado: adopcion_por_industria.png")
    plt.show()

# CORRECCIÓN 4: Exportación para Power BI (CON NOMBRES CORRECTOS)
print("\n📤 EXPORTANDO DATOS PARA POWER BI...")

# Asegurar que existe la carpeta data
if not os.path.exists('data'):
    os.makedirs('data')
    print("📁 Carpeta 'data/' creada")

# Exportar dataset limpio
df_clean.to_csv('data/ai_adoption_cleaned.csv', index=False)
print("✅ Dataset limpio exportado: data/ai_adoption_cleaned.csv")

# Crear datasets resumidos para el dashboard
try:
    # CORRECCIÓN: Usar nombres de columnas en minúsculas
    # Análisis por industria (si existe la columna)
    if 'industry' in df_clean.columns:
        industry_summary = df_clean.groupby('industry').agg({
            'adoption_rate': 'mean',
            'daily_active_users': 'mean'
        }).round(3)
        
        # Agregar conteo de empresas por industria
        industry_summary['company_count'] = df_clean.groupby('industry').size()
        
        industry_summary.to_csv('data/industry_summary.csv')
        print("✅ Resumen por industria exportado: data/industry_summary.csv")
    
    # Análisis por país/región (si existe)
    if 'country' in df_clean.columns:
        country_summary = df_clean.groupby('country').agg({
            'adoption_rate': 'mean',
            'daily_active_users': 'mean'
        }).round(3)
        country_summary['company_count'] = df_clean.groupby('country').size()
        country_summary.to_csv('data/country_summary.csv')
        print("✅ Resumen por país exportado: data/country_summary.csv")
    
    # Dataset para tendencias temporales (usando 'year')
    if 'year' in df_clean.columns:
        yearly_trends = df_clean.groupby('year').agg({
            'adoption_rate': 'mean',
            'daily_active_users': 'mean'
        }).round(3)
        yearly_trends['company_count'] = df_clean.groupby('year').size()
        yearly_trends.to_csv('data/yearly_trends.csv')
        print("✅ Tendencias anuales exportadas: data/yearly_trends.csv")
    
    # Top 10 análisis (para gráficos específicos)
    if 'adoption_rate' in df_clean.columns and 'industry' in df_clean.columns:
        # Top 10 industrias con mayor adopción
        top_industries = df_clean.groupby('industry')['adoption_rate'].mean().nlargest(10).reset_index()
        top_industries.to_csv('data/top_10_industries.csv', index=False)
        print("✅ Top 10 industrias exportado: data/top_10_industries.csv")
    
    print("\n🎉 Todos los datos exportados exitosamente!")
    print("📊 Archivos listos en carpeta 'data/':")
    print("   - ai_adoption_cleaned.csv (dataset completo)")
    print("   - industry_summary.csv (resumen por industria)")
    print("   - country_summary.csv (resumen por país)")
    print("   - yearly_trends.csv (tendencias anuales)")
    print("   - top_10_industries.csv (top 10 industrias)")

except Exception as e:
    print(f"❌ Error en exportación: {e}")
    import traceback
    print(f"🔍 Detalles: {traceback.format_exc()}")

print("\n🚀 Análisis Exploratorio COMPLETADO!")