import pandas as pd
import numpy as np
import random
from datetime import timedelta

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
