import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Any
import logging
import time
import traceback

logger = logging.getLogger(__name__)

from australia_climate_analysis.pipelines.regression_model.nodes import (
    split_data_for_regression
)

# ---------- ENTRENAMIENTO ----------
def train_multiple_regression_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena un modelo de regresión lineal múltiple."""
    logger.info(f"Iniciando entrenamiento de modelo de regresión múltiple con {X_train.shape[0]} muestras y {X_train.shape[1]} características")
    
    start_time = time.time()
    
    # Configurar el modelo con parámetros eficientes
    model = LinearRegression(n_jobs=-1)  # Usar todos los núcleos disponibles
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
    logger.info(f"Coeficientes: {model.coef_}")
    logger.info(f"Intercepto: {model.intercept_}")
    
    return model

# ---------- EVALUACIÓN ----------
def evaluate_multiple_regression_model(model, X_test, y_test, run_id: str):
    """Evalúa el modelo de regresión múltiple y devuelve métricas."""
    # Crear un identificador único para este modelo si no se proporciona
    if not run_id or run_id == "":
        run_id = f"multiple_reg_{time.strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"No se proporcionó run_id, usando: {run_id}")
    else:
        logger.info(f"Usando run_id proporcionado: {run_id}")
    
    # Medir tiempo de predicción
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calcular métricas básicas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Evaluación completada en {prediction_time:.2f} segundos")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"R²: {r2:.4f}")
    
    # Crear JSON para archivo local
    metrics_json = {
        "model_name": "MultipleRegression",
        "mse": float(mse),
        "r2": float(r2),
        "prediction_time_seconds": float(prediction_time)
    }
    
    # Crear DataFrame para PostgreSQL - versión minimalista
    metrics_long = pd.DataFrame([
        {"run_id": str(run_id), "model_name": "MultipleRegression", "metric": "mse", "value": float(mse)},
        {"run_id": str(run_id), "model_name": "MultipleRegression", "metric": "r2", "value": float(r2)},
        {"run_id": str(run_id), "model_name": "MultipleRegression", "metric": "prediction_time_seconds", "value": float(prediction_time)}
    ])
    
    return metrics_json, metrics_long