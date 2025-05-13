import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

from australia_climate_analysis.pipelines.regression_model.nodes import (
    split_data_for_regression
)

# ---------- ENTRENAMIENTO ----------
def train_svm_regression_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena un modelo SVM para regresión con configuración optimizada para velocidad."""
    # Usar un kernel lineal que es mucho más rápido que RBF
    model = SVR(kernel='linear', C=1.0, epsilon=0.1)
    
    # Opcional: Usar una muestra más pequeña para entrenamiento más rápido
    # Si el dataset es muy grande, usar solo el 30% de los datos
    if X_train.shape[0] > 50000:
        sample_size = int(X_train.shape[0] * 0.3)
        # Asegurar reproducibilidad con random_state
        sample_indices = np.random.RandomState(42).choice(X_train.index, size=sample_size, replace=False)
        X_train_sample = X_train.loc[sample_indices]
        y_train_sample = y_train.loc[sample_indices]
        print(f"Usando muestra de {sample_size} registros (30% del total) para entrenamiento rápido")
        model.fit(X_train_sample, y_train_sample)
    else:
        model.fit(X_train, y_train)
    
    return model

# ---------- EVALUACIÓN ----------
def evaluate_svm_regression_model(model, X_test, y_test, run_id: str):
    """Evalúa el modelo SVM optimizado y devuelve métricas."""
    import time
    
    # Medir tiempo de predicción
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Modelo SVM optimizado - MSE: {mse:.4f}, R²: {r2:.4f}")
    logger.info(f"Tiempo de predicción para {len(X_test)} muestras: {prediction_time:.2f} segundos")
    
    metrics = {
        "mse": mse,
        "r2": r2,
        "prediction_time_seconds": prediction_time
    }

    # ――― JSON plano para artefacto local ----------
    metrics_json = {
        "model_name": "SVR",
        **metrics
    }

    # ――― Formato "largo" para la tabla ------------
    metrics_long = pd.DataFrame([
        {
            "run_id": run_id,
            "model_name": "SVR",
            "metric": k,
            "value": v,
        }
        for k, v in metrics.items()
    ])

    return metrics_json, metrics_long