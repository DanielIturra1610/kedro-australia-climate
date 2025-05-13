import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def cargar_datos(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Carga los datos del clima de Australia."""
    logger.info(f"Datos cargados con forma: {raw_data.shape}")
    return raw_data

def mostrar_valores_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Muestra los valores nulos en el dataset."""
    nulos = df.isnull().sum()
    logger.info(f"Columnas con valores nulos: {nulos[nulos > 0]}")
    return df

def imputar_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores numéricos faltantes con la mediana."""
    # Identificar columnas numéricas
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Imputar con la mediana
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    logger.info(f"Imputación numérica completada para {len(num_cols)} columnas")
    return df

def imputar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores categóricos faltantes con la moda."""
    # Identificar columnas categóricas
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Imputar con la moda
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    logger.info(f"Imputación categórica completada para {len(cat_cols)} columnas")
    return df

def convertir_fecha(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la columna Date a formato datetime."""
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        logger.info("Columna Date convertida a datetime")
    return df

def outliers_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica y trata outliers en la columna Rainfall."""
    if 'Rainfall' in df.columns:
        # Calcular límites para outliers (método IQR)
        Q1 = df['Rainfall'].quantile(0.25)
        Q3 = df['Rainfall'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identificar outliers
        outliers = df[(df['Rainfall'] < lower_bound) | (df['Rainfall'] > upper_bound)].shape[0]
        logger.info(f"Identificados {outliers} outliers en Rainfall")
        
        # Opción: recortar outliers extremos
        df.loc[df['Rainfall'] > upper_bound, 'Rainfall'] = upper_bound
        
    return df

def agregar_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega características temporales basadas en la fecha."""
    if 'Date' in df.columns:
        # Extraer componentes de fecha
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Agregar estación (para hemisferio sur)
        conditions = [
            (df['Month'] >= 3) & (df['Month'] <= 5),  # Otoño
            (df['Month'] >= 6) & (df['Month'] <= 8),  # Invierno
            (df['Month'] >= 9) & (df['Month'] <= 11), # Primavera
            (df['Month'] == 12) | (df['Month'] <= 2)  # Verano
        ]
        seasons = ['Autumn', 'Winter', 'Spring', 'Summer']
        df['Season'] = np.select(conditions, seasons, default='Unknown')
        
        logger.info("Características temporales agregadas")
    
    # Seleccionar columnas relevantes para el análisis de riesgo climático
    features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm',
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am',
        'Temp3pm', 'RainToday', 'Year', 'Month', 'Season'
    ]
    
    # Filtrar solo las columnas que existen en el DataFrame
    available_features = [col for col in features if col in df.columns]
    df_features = df[available_features].copy()
    
    logger.info(f"Dataset final con {df_features.shape[0]} filas y {df_features.shape[1]} columnas")
    
    return df_features