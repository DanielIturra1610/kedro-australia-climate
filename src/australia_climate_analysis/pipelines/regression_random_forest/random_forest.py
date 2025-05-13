from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# ---------- ENTRENAMIENTO DEL MODELO ----------
def train_random_forest_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena un modelo de Random Forest para regresión."""
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

# ---------- EVALUACIÓN DEL MODELO ----------
def evaluate_random_forest_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evalúa el modelo de Random Forest usando MSE y R²."""
    y_pred = model.predict(X_test)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    # Guardar métricas en formato JSON (local)
    metrics_json = {
        "model_name": type(model).__name__,
        **metrics
    }

    # Formato "largo" para la base de datos o tabla (si se usa)
    metrics_long = pd.DataFrame([{
        "model_name": type(model).__name__,
        "metric": k,
        "value": v,
    } for k, v in metrics.items()])

    return metrics_json, metrics_long
