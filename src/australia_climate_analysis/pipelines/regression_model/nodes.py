import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import datetime

logger = logging.getLogger(__name__)


def split_data_for_regression(
    weather_data: pd.DataFrame,
    test_size: float,
    random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepara los datos de entrada para el modelo de regresión."""
    weather_data = weather_data.loc[:, ~weather_data.columns.duplicated()]
    logger.info("[SPLIT] Columnas duplicadas eliminadas: %s", weather_data.columns.duplicated())

    logger.info("[SPLIT] Shape original de datos: %s", weather_data.shape)

    # Seleccionar variables relevantes para regresión
    features = ["MinTemp", "Rainfall", "Humidity3pm", "WindGustSpeed"]
    target = "MaxTemp"

    # Nos quedamos solo con esas columnas
    df = weather_data[features + [target]].copy()

    logger.info("[SPLIT] Shape después de seleccionar columnas: %s", df.shape)

    # Imputar valores faltantes (usamos mediana para robustez)
    df = df.fillna(df.median())

    logger.info("[SPLIT] Shape después de imputar nulos: %s", df.shape)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info("[SPLIT] Shape de X_train: %s, X_test: %s", X_train.shape, X_test.shape)

    return X_train, X_test, y_train.to_frame(name="target"), y_test.to_frame(name="target")

def train_regression_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena el modelo de regresión lineal."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_regression_model(model, X_test, y_test, run_id: str):
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


def save_model(model) -> None:
    """Guarda el modelo entrenado."""
    joblib.dump(model, "data/06_models/regression_model.pkl")
