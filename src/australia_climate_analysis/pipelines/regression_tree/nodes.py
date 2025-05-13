# solo cambian las importaciones y la función de entrenamiento
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import logging, datetime

logger = logging.getLogger(__name__)

from australia_climate_analysis.pipelines.regression_model.nodes import (
    split_data_for_regression
)

# ---------- ENTRENAMIENTO ----------
def train_regression_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena un Árbol de Decisión para regresión."""
    model = DecisionTreeRegressor(
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# ---------- EVALUACIÓN ----------
def evaluate_regression_model_tree(model, X_test, y_test, run_id: str):
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score

    y_pred = model.predict(X_test)
    metrics = {
        "mse":  mean_squared_error(y_test, y_pred),
        "r2":   r2_score(y_test, y_pred),
    }

    # ――― JSON plano para artefacto local ----------
    metrics_json = {
        "model_name": type(model).__name__,
        **metrics
    }

    # ――― Formato “largo” para la tabla ------------
    metrics_long = pd.DataFrame([
        {
            "run_id": run_id,
            "model_name": type(model).__name__,
            "metric": k,
            "value": v,
        }
        for k, v in metrics.items()
    ])

    return metrics_json, metrics_long



