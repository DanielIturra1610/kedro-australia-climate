import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Tuple, Any

from australia_climate_analysis.pipelines.regression_model.nodes import (
    split_data_for_regression
)

# ---------- ENTRENAMIENTO ----------
def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena un modelo Random Forest para regresión."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# ---------- EVALUACIÓN ----------
def evaluate_random_forest_model(model, X_test, y_test, run_id: str):
    """Evalúa el modelo Random Forest y devuelve métricas."""
    y_pred = model.predict(X_test)
    metrics = {
        "mse":  mean_squared_error(y_test, y_pred),
        "r2":   r2_score(y_test, y_pred),
    }

    # ――― JSON plano para artefacto local ----------
    metrics_json = {
        "model_name": "RandomForestRegressor",
        **metrics
    }

    # ――― Formato "largo" para la tabla ------------
    metrics_long = pd.DataFrame([
        {
            "run_id": run_id,
            "model_name": "RandomForestRegressor",
            "metric": k,
            "value": v,
        }
        for k, v in metrics.items()
    ])

    return metrics_json, metrics_long