"""
Nodos para el pipeline de Naive Bayes para predecir el clima de mañana basado en el clima de hoy.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, Tuple, Any, List
import logging
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def prepare_data_for_naive_bayes(
    weather_data: pd.DataFrame, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepara los datos para el modelo de Naive Bayes, creando características para predecir
    el clima del día siguiente basado en el clima actual.
    
    Args:
        weather_data: DataFrame con datos meteorológicos originales sin procesar
        test_size: Proporción de datos para prueba
        random_state: Semilla para reproducibilidad
    
    Returns:
        X_train: Características de entrenamiento
        X_test: Características de prueba
        y_train: Etiquetas de entrenamiento (RainTomorrow)
        y_test: Etiquetas de prueba (RainTomorrow)
    """
    logger.info(f"Preparando datos para modelo Naive Bayes con {len(weather_data)} registros")
    
    # Convertir la columna Date a datetime si no lo es ya
    if weather_data['Date'].dtype != 'datetime64[ns]':
        weather_data['Date'] = pd.to_datetime(weather_data['Date'])
    
    # Ordenar datos por fecha y ubicación
    weather_data = weather_data.sort_values(by=['Location', 'Date'])
    
    # Crear variable objetivo: ¿Lloverá mañana?
    # Primero creamos un identificador único para cada ubicación y fecha
    weather_data['Location_Date'] = weather_data['Location'] + '_' + weather_data['Date'].dt.strftime('%Y-%m-%d')
    
    # Crear un DataFrame con la información del día siguiente
    next_day_rain = weather_data[['Location_Date', 'RainToday']].copy()
    next_day_rain.columns = ['Location_Date', 'RainTomorrow']
    
    # Calcular la fecha del día siguiente para cada registro
    weather_data['NextDate'] = (weather_data['Date'] + timedelta(days=1)).dt.strftime('%Y-%m-%d')
    weather_data['Next_Location_Date'] = weather_data['Location'] + '_' + weather_data['NextDate']
    
    # Unir con la información de lluvia del día siguiente
    weather_data = weather_data.merge(
        next_day_rain,
        left_on='Next_Location_Date',
        right_on='Location_Date',
        how='left',
        suffixes=('', '_next')
    )
    
    # Eliminar filas donde no tenemos información del día siguiente
    weather_data = weather_data.dropna(subset=['RainTomorrow'])
    
    logger.info(f"Después de preparar los datos para secuencia temporal: {len(weather_data)} registros")
    
    # Seleccionar características relevantes para predecir la lluvia
    features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
        'RainToday'
    ]
    
    # Verificar qué columnas están disponibles en el dataset
    available_features = [col for col in features if col in weather_data.columns]
    logger.info(f"Características disponibles: {available_features}")
    
    # Filtrar solo las columnas necesarias que están disponibles
    X = weather_data[available_features].copy()
    
    # Convertir variables categóricas a numéricas
    if 'RainToday' in X.columns:
        X['RainToday'] = X['RainToday'].map({'Yes': 1, 'No': 0})
    
    # Imputar valores faltantes con la media
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            X[col] = X[col].fillna(X[col].mean())
        else:
            # Para columnas no numéricas, intentar convertir a numérico si es posible
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].mean())
            except:
                # Si no se puede convertir, eliminar la columna
                logger.warning(f"Columna {col} no se puede convertir a numérico y será eliminada")
                X = X.drop(columns=[col])
    
    # Convertir la variable objetivo a formato numérico
    y = weather_data['RainTomorrow'].map({'Yes': 1, 'No': 0})
    
    # Dividir en conjuntos de entrenamiento y prueba
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    logger.info(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras, Conjunto de prueba: {X_test.shape[0]} muestras")
    logger.info(f"Distribución de clases en entrenamiento - No lluvia: {(y_train == 0).sum()}, Lluvia: {(y_train == 1).sum()}")
    
    return X_train, X_test, y_train, y_test

def train_naive_bayes_model(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Entrena un modelo de Naive Bayes para predecir si lloverá mañana.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Etiquetas de entrenamiento (RainTomorrow)
    
    Returns:
        Modelo entrenado de Naive Bayes
    """
    logger.info(f"Entrenando modelo Naive Bayes con {X_train.shape[0]} muestras y {X_train.shape[1]} características")
    
    start_time = time.time()
    
    # Crear un pipeline con preprocesamiento y el modelo Naive Bayes
    # Separar columnas numéricas y categóricas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    # Crear el pipeline completo
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    logger.info(f"Entrenamiento completado en {training_time:.2f} segundos")
    
    # Obtener probabilidades de clase para algunas muestras
    class_probs = model.predict_proba(X_train.iloc[:5])
    logger.info(f"Probabilidades de clase para las primeras 5 muestras:\n{class_probs}")
    
    return model

def evaluate_naive_bayes_model(model, X_test, y_test, run_id: str):
    """
    Evalúa el modelo de Naive Bayes y devuelve métricas.
    
    Args:
        model: Modelo entrenado de Naive Bayes
        X_test: Características de prueba
        y_test: Etiquetas de prueba (RainTomorrow)
        run_id: Identificador de la ejecución
    
    Returns:
        Métricas de evaluación en formato JSON y DataFrame para PostgreSQL
    """
    # Crear un identificador único para este modelo si no se proporciona
    if not run_id or run_id == "":
        run_id = f"naive_bayes_{time.strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"No se proporcionó run_id, usando: {run_id}")
    else:
        logger.info(f"Usando run_id proporcionado: {run_id}")
    
    # Medir tiempo de predicción
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # Obtener el reporte de clasificación
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Evaluación completada en {prediction_time:.2f} segundos")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Matriz de confusión:\n{cm}")
    
    # Extraer métricas específicas del reporte
    precision_no_rain = report.get('0', {}).get('precision', 0)
    recall_no_rain = report.get('0', {}).get('recall', 0)
    precision_rain = report.get('1', {}).get('precision', 0)
    recall_rain = report.get('1', {}).get('recall', 0)
    f1_no_rain = report.get('0', {}).get('f1-score', 0)
    f1_rain = report.get('1', {}).get('f1-score', 0)
    
    # Crear JSON para archivo local
    metrics_json = {
        "model_name": "NaiveBayes",
        "accuracy": float(accuracy),
        "precision_no_rain": float(precision_no_rain),
        "recall_no_rain": float(recall_no_rain),
        "precision_rain": float(precision_rain),
        "recall_rain": float(recall_rain),
        "f1_no_rain": float(f1_no_rain),
        "f1_rain": float(f1_rain),
        "prediction_time_seconds": float(prediction_time)
    }
    
    # Crear DataFrame para PostgreSQL
    metrics_long = pd.DataFrame([
        {"run_id": str(run_id), "model_name": "NaiveBayes", "metric": "accuracy", "value": float(accuracy)},
        {"run_id": str(run_id), "model_name": "NaiveBayes", "metric": "precision_no_rain", "value": float(precision_no_rain)},
        {"run_id": str(run_id), "model_name": "NaiveBayes", "metric": "recall_no_rain", "value": float(recall_no_rain)},
        {"run_id": str(run_id), "model_name": "NaiveBayes", "metric": "precision_rain", "value": float(precision_rain)},
        {"run_id": str(run_id), "model_name": "NaiveBayes", "metric": "recall_rain", "value": float(recall_rain)},
        {"run_id": str(run_id), "model_name": "NaiveBayes", "metric": "f1_no_rain", "value": float(f1_no_rain)},
        {"run_id": str(run_id), "model_name": "NaiveBayes", "metric": "f1_rain", "value": float(f1_rain)},
        {"run_id": str(run_id), "model_name": "NaiveBayes", "metric": "prediction_time_seconds", "value": float(prediction_time)}
    ])
    
    return metrics_json, metrics_long

def predict_next_day_weather(model, weather_data: pd.DataFrame, prediction_date: str = None):
    """
    Predice si lloverá mañana basado en los datos meteorológicos de hoy.
    
    Args:
        model: Modelo entrenado de Naive Bayes
        weather_data: DataFrame con datos meteorológicos originales sin procesar
        prediction_date: Fecha para la que se quiere hacer la predicción (formato YYYY-MM-DD)
                         Si es None, se usa la fecha más reciente en los datos
    
    Returns:
        DataFrame con las predicciones para cada ubicación
    """
    logger.info("Preparando datos para predicción del clima de mañana")
    
    # Convertir la columna Date a datetime si no lo es ya
    if weather_data['Date'].dtype != 'datetime64[ns]':
        weather_data['Date'] = pd.to_datetime(weather_data['Date'])
    
    # Si no se proporciona fecha, usar la más reciente
    if prediction_date is None:
        prediction_date = weather_data['Date'].max().strftime('%Y-%m-%d')
        logger.info(f"No se proporcionó fecha de predicción, usando la más reciente: {prediction_date}")
    else:
        # Convertir prediction_date a datetime para comparación
        prediction_date = pd.to_datetime(prediction_date).strftime('%Y-%m-%d')
    
    # Filtrar datos para la fecha de predicción
    today_data = weather_data[weather_data['Date'].dt.strftime('%Y-%m-%d') == prediction_date].copy()
    
    if len(today_data) == 0:
        logger.warning(f"No hay datos para la fecha {prediction_date}")
        return pd.DataFrame({"error": [f"No hay datos para la fecha {prediction_date}"]})
    
    # Seleccionar características relevantes
    features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
        'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm',
        'RainToday'
    ]
    
    # Verificar qué columnas están disponibles en el dataset
    available_features = [col for col in features if col in today_data.columns]
    logger.info(f"Características disponibles para predicción: {available_features}")
    
    # Preparar datos para predicción
    X_pred = today_data[available_features].copy()
    
    # Convertir variables categóricas a numéricas
    if 'RainToday' in X_pred.columns:
        X_pred['RainToday'] = X_pred['RainToday'].map({'Yes': 1, 'No': 0})
    
    # Imputar valores faltantes con la media
    for col in X_pred.columns:
        if X_pred[col].dtype in [np.float64, np.int64]:
            X_pred[col] = X_pred[col].fillna(X_pred[col].mean())
        else:
            # Para columnas no numéricas, intentar convertir a numérico si es posible
            try:
                X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')
                X_pred[col] = X_pred[col].fillna(X_pred[col].mean())
            except:
                # Si no se puede convertir, eliminar la columna
                logger.warning(f"Columna {col} no se puede convertir a numérico y será eliminada")
                X_pred = X_pred.drop(columns=[col])
    
    # Verificar que tenemos las mismas columnas que el modelo espera
    model_features = model.named_steps['preprocessor'].get_feature_names_out()
    logger.info(f"Características esperadas por el modelo: {model_features}")
    
    # Realizar predicción
    try:
        predictions = model.predict(X_pred)
        probabilities = model.predict_proba(X_pred)
        
        # Crear DataFrame con resultados
        results = today_data[['Location', 'Date']].copy()
        results['NextDate'] = (results['Date'] + timedelta(days=1)).dt.strftime('%Y-%m-%d')
        results['RainTomorrow_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
        results['RainTomorrow_Probability'] = [prob[1] for prob in probabilities]  # Probabilidad de lluvia
        
        logger.info(f"Predicciones realizadas para {len(results)} ubicaciones")
        logger.info(f"Distribución de predicciones - No lluvia: {(predictions == 0).sum()}, Lluvia: {(predictions == 1).sum()}")
        
        return results
    except Exception as e:
        logger.error(f"Error al realizar predicciones: {str(e)}")
        return pd.DataFrame({"error": [f"Error al realizar predicciones: {str(e)}"]})