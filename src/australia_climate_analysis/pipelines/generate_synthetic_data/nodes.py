import pandas as pd
import numpy as np
import random
from datetime import timedelta
from typing import List, Dict, Any, Tuple

def generate_synthetic_day(df: pd.DataFrame) -> pd.DataFrame:
    sample = df.sample(n=1).copy()

    # Columnas numéricas para aplicar ruido
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if pd.notnull(sample[col].values[0]):
            std = df[col].std() * 0.1  # 10% del std dev original
            sample[col] = sample[col] + np.random.normal(0, std)

    return sample.reset_index(drop=True)


def generate_multiple_synthetic_days(df: pd.DataFrame, n_days: int = 10, start_date: str = "2025-01-01") -> pd.DataFrame:
    # Convertir columna Date a datetime por si acaso
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Filtrar solo filas que coincidan con el mes del start_date
    start = pd.to_datetime(start_date)
    df_month_filtered = df[df["Date"].dt.month == start.month].copy()

    # Validar que haya suficientes datos para muestreo
    if df_month_filtered.empty:
        raise ValueError(f"No hay datos para el mes {start.month}. Por favor verifica el dataset original.")

    synthetic_days = []

    for i in range(n_days):
        day = generate_synthetic_day(df_month_filtered)
        day["Date"] = start + timedelta(days=i)

        # Completar columnas categóricas si hicieran falta
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col not in day.columns or pd.isnull(day[col].values[0]):
                day[col] = df[col].dropna().sample(n=1).values[0]

        synthetic_days.append(day)

    return pd.concat(synthetic_days, ignore_index=True)


def generate_coherent_sequence(df: pd.DataFrame, location: str, start_date: str, sequence_length: int = 3) -> pd.DataFrame:
    """
    Genera una secuencia coherente de días con tendencia realista para una ubicación específica.
    
    Args:
        df: DataFrame con datos históricos
        location: Ubicación para la cual generar datos
        start_date: Fecha de inicio para la secuencia
        sequence_length: Longitud de la secuencia (por defecto 3 días)
        
    Returns:
        DataFrame con la secuencia de días generada
    """
    # Convertir a datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    start = pd.to_datetime(start_date)
    
    # Filtrar por ubicación y mes similar
    df_filtered = df[(df["Location"] == location) & (df["Date"].dt.month == start.month)].copy()
    
    if len(df_filtered) < 10:
        # Si no hay suficientes datos, usar todos los datos de esa ubicación
        df_filtered = df[df["Location"] == location].copy()
        
    if df_filtered.empty:
        raise ValueError(f"No hay datos suficientes para la ubicación {location}")
    
    # Variables que deben mantener coherencia día a día
    coherent_vars = ["MinTemp", "MaxTemp", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"]
    
    # Obtener un punto de partida
    base_day = generate_synthetic_day(df_filtered)
    sequence = [base_day.copy()]
    
    # Generar días siguientes con tendencias coherentes
    for i in range(1, sequence_length):
        next_day = generate_synthetic_day(df_filtered)
        
        # Mantener coherencia en variables clave
        for var in coherent_vars:
            if var in base_day.columns and var in next_day.columns:
                # Calcular tendencia basada en datos históricos
                # Buscar cambios típicos día a día para esta variable
                daily_changes = []
                for j in range(1, len(df_filtered)):
                    if pd.notnull(df_filtered[var].iloc[j]) and pd.notnull(df_filtered[var].iloc[j-1]):
                        daily_changes.append(df_filtered[var].iloc[j] - df_filtered[var].iloc[j-1])
                
                if daily_changes:
                    # Calcular cambio promedio y desviación estándar
                    mean_change = np.mean(daily_changes)
                    std_change = np.std(daily_changes) or 0.5  # Evitar std = 0
                    
                    # Aplicar un cambio coherente al día anterior
                    prev_value = sequence[i-1][var].values[0]
                    realistic_change = np.random.normal(mean_change, std_change * 0.8)
                    next_day[var] = prev_value + realistic_change
                    
        # Actualizar fecha
        next_day["Date"] = start + timedelta(days=i)
        
        # Mantener Location constante
        next_day["Location"] = location
        
        # Para RainToday/RainTomorrow, asegurar coherencia
        if i > 0 and "RainTomorrow" in sequence[i-1].columns and "RainToday" in next_day.columns:
            next_day["RainToday"] = sequence[i-1]["RainTomorrow"].values[0]
        
        sequence.append(next_day)
    
    return pd.concat(sequence, ignore_index=True)


def generate_forecast_sequences(df: pd.DataFrame, locations: List[str] = None, 
                               start_date: str = None, sequence_length: int = 3) -> pd.DataFrame:
    """
    Genera secuencias de predicción para múltiples ubicaciones.
    
    Args:
        df: DataFrame con datos históricos
        locations: Lista de ubicaciones (si es None, se usan todas las disponibles)
        start_date: Fecha de inicio (si es None, se usa la fecha actual)
        sequence_length: Longitud de la secuencia
        
    Returns:
        DataFrame con las secuencias generadas para todas las ubicaciones
    """
    from datetime import datetime
    
    # Si no se proporciona fecha, usar fecha actual
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")
    
    # Si no se proporcionan ubicaciones, usar todas las disponibles
    if locations is None:
        locations = df["Location"].unique().tolist()
    
    all_sequences = []
    
    for location in locations:
        try:
            sequence = generate_coherent_sequence(
                df, 
                location=location,
                start_date=start_date, 
                sequence_length=sequence_length
            )
            all_sequences.append(sequence)
        except ValueError as e:
            print(f"Error generando datos para {location}: {e}")
            continue
    
    if not all_sequences:
        raise ValueError("No se pudieron generar secuencias para ninguna ubicación")
    
    return pd.concat(all_sequences, ignore_index=True)