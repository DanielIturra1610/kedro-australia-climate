from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ---------- ESCALADO DE LOS DATOS ----------
def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Estandariza las características de los datos para SVM."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ---------- ENTRENAMIENTO DEL MODELO ----------
def train_svm_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena un modelo de Support Vector Machine para regresión."""
    # Asegúrate de aplanar y_train
    y_train = y_train.values.ravel()

    # Escalado de los datos
    X_train_scaled, _ = scale_data(X_train, X_train)  # Solo usamos X_train para escalar

    # Creación y entrenamiento del modelo
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    return model

# ---------- EVALUACIÓN DEL MODELO ----------
def evaluate_svm_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evalúa el modelo de SVM usando MSE y R²."""
    # Asegúrate de aplanar y_test
    y_test = y_test.values.ravel()

    # Escalado de los datos de prueba
    _, X_test_scaled = scale_data(X_test, X_test)  # Escalamos X_test usando el scaler entrenado

    # Realizamos las predicciones
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    # Formato JSON plano para el artefacto local
    metrics_json = {
        "model_name": type(model).__name__,
        **metrics
    }

    # Formato "largo" para la tabla
    metrics_long = pd.DataFrame([{
        "model_name": type(model).__name__,
        "metric": k,
        "value": v,
    } for k, v in metrics.items()])

    return metrics_json, metrics_long
