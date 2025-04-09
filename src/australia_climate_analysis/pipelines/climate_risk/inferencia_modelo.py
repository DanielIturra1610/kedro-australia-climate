import pandas as pd
from sklearn.base import BaseEstimator

def inferir_riesgo_climatico(model: BaseEstimator, datos: pd.DataFrame) -> pd.DataFrame:
    """Realiza inferencia con el modelo de clasificación de riesgo climático.

    Args:
        model: Modelo de clasificación previamente entrenado.
        datos: DataFrame con las características para predecir.

    Returns:
        DataFrame con las predicciones de riesgo por ciudad y fecha.
    """
    predicciones = model.predict(datos.drop(columns=["Location", "Date"]))
    resultados = datos[["Location", "Date"]].copy()
    resultados["Riesgo_Predicho"] = predicciones
    return resultados
