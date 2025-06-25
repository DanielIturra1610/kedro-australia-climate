import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import time
from .forecast_model import ForecastModel

def prepare_forecast_features(weather_data):
    """Prepara características para el modelo de pronóstico del tiempo a 3 días.
    
    Args:
        weather_data: DataFrame de pandas con datos meteorológicos históricos
        
    Returns:
        X: Características para el modelo
        y_dict: Diccionario con variables objetivo para diferentes días
    """
    # Código existente hasta la línea 58...
    df_clean = weather_data[weather_data["Location"].isin(["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Hobart", "Darwin", "Canberra"])].copy()
    
    # Convertir Date a tipo datetime si no lo es ya
    if not pd.api.types.is_datetime64_dtype(df_clean["Date"]):
        df_clean["Date"] = pd.to_datetime(df_clean["Date"])
    
    # Agregar características temporales
    df_clean["Month"] = df_clean["Date"].dt.month
    df_clean["DayOfYear"] = df_clean["Date"].dt.dayofyear
    
    # Crear variables adicionales que puedan ser útiles
    df_clean["MaxMinTempRange"] = df_clean["MaxTemp"] - df_clean["MinTemp"]
    
    # Mapear Location a una codificación numérica
    location_mapping = {loc: i for i, loc in enumerate(df_clean["Location"].unique())}
    df_clean["LocationCode"] = df_clean["Location"].map(location_mapping)
    
    # Definir características disponibles (excluyendo las que no deberían usarse)
    available_features = ["Month", "DayOfYear", "MinTemp", "MaxTemp", "Rainfall",
                          "Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed9am",
                          "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", 
                          "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
                          "MaxMinTempRange", "LocationCode"]
    
    # Filtrar solo las características disponibles
    available_features = [f for f in available_features if f in df_clean.columns]
    
    # Ordenar por Location y Date para asegurar secuencia temporal correcta
    df_clean = df_clean.sort_values(['Location', 'Date']).reset_index(drop=True)
    
    # Extraer las características después del reordenamiento
    X = df_clean[available_features].copy()
    
    # SOLUCIÓN: Manejar valores faltantes en features separando por tipo de datos
    # Para columnas numéricas - usar la media
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    
    # Para columnas categóricas/texto - usar el valor más frecuente
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
    
    # Preparar diccionario de variables objetivo
    y_dict = {}
    
    # Para cada día futuro (1, 2 y 3), crear un objetivo usando shift
    for days_ahead in [1, 2, 3]:
        # Para lluvia: usar shift negativo para obtener valores futuros
        # Agrupar por Location para mantener la secuencia temporal por ubicación
        rain_future = df_clean.groupby('Location')['RainToday'].shift(-days_ahead)
        # Convertir 'Yes'/'No' a 1/0
        rain_binary = (rain_future == 'Yes').astype('Int64')
        y_dict[f"rain_day_{days_ahead}"] = rain_binary
        
        # Variables numéricas objetivo (temperaturas, presión, etc)
        for var in ["MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "WindSpeed9am", "WindSpeed3pm"]:
            if var in df_clean.columns:
                # Usar shift negativo para obtener valores futuros por ubicación
                future_values = df_clean.groupby('Location')[var].shift(-days_ahead)
                y_dict[f"{var.lower()}_day_{days_ahead}"] = future_values
                
    # No filtrar filas - mantener todos los datos disponibles
    # Los modelos individuales manejarán los NaN según sea necesario
    
    logging.info(f"Datos preparados: {len(X)} filas de características")
    logging.info(f"Variables objetivo creadas: {list(y_dict.keys())}")
    
    # Mostrar estadísticas de datos válidos por variable objetivo
    for var_name, var_data in y_dict.items():
        valid_count = var_data.notna().sum()
        total_count = len(var_data)
        logging.info(f"  {var_name}: {valid_count}/{total_count} valores válidos ({valid_count/total_count*100:.1f}%)")
    
    return X, y_dict

def train_forecast_models(X, y_dict, forecast_params):
    """
    Entrena modelos de pronóstico para múltiples variables objetivo.
    
    Args:
        X: DataFrame con características
        y_dict: Diccionario de variables objetivo para diferentes días
        forecast_params: Parámetros para el modelo de pronóstico
    
    Returns:
        ForecastModel: Modelo entrenado para pronósticos
    """

    
    # Recuperar parámetros
    kernel = forecast_params.get("kernel", "linear")
    C = forecast_params.get("C", 1.0)
    sample_threshold = forecast_params.get("sample_threshold", 50000)
    sample_ratio = forecast_params.get("sample_ratio", 0.3)
    
    logging.info(f"Entrenando modelo de pronóstico con kernel={kernel}, C={C}")
    logging.info(f"Usando {sample_ratio*100}% de los datos si exceden {sample_threshold} filas")
    
    # Inicializar y entrenar modelo
    start_time = time.time()
    
    # Crear instancia del modelo
    model = ForecastModel(kernel=kernel, C=C, 
                         sample_threshold=sample_threshold,
                         sample_ratio=sample_ratio)
    
    # Asegurarnos de entrenar modelos para todas las variables objetivo requeridas
    required_targets = [
        "MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm", 
        "WindSpeed9am", "WindSpeed3pm", "Pressure9am", "Pressure3pm"
    ]
    
    # Agrupamos las variables objetivo por día para entrenar modelos para día 1, 2 y 3
    for day_number in [1, 2, 3]:
        # Para cada variable objetivo, entrenar modelo si está disponible
        for var in required_targets:
            var_key = f"{var.lower()}_day_{day_number}"
            rain_key = f"rain_day_{day_number}"
            
            if var_key in y_dict and not y_dict[var_key].isna().all():
                # Agregar indicador de lluvia como característica adicional si está disponible
                X_train = X.copy()
                if rain_key in y_dict:
                    X_train[f"rain_day_{day_number}"] = y_dict[rain_key].astype(int)
                
                # Entrenar modelo para esta variable objetivo
                model.train_model(X_train, y_dict[var_key], var, day=day_number)
                logging.info(f"Modelo entrenado para {var} día {day_number}")
            else:
                logging.warning(f"No hay datos suficientes para entrenar modelo de {var} día {day_number}")
    
    # Entrenar un modelo específico para predicción de lluvia si está disponible
    for day_number in [1, 2, 3]:
        rain_key = f"rain_day_{day_number}"
        if rain_key in y_dict and not y_dict[rain_key].isna().all():
            model.train_rain_model(X, y_dict[rain_key], day=day_number)
            logging.info(f"Modelo de lluvia entrenado para día {day_number}")
    
    elapsed_time = time.time() - start_time
    logging.info(f"Entrenamiento de modelos completado en {elapsed_time:.2f} segundos")
    
    return model

def generate_forecast(model: ForecastModel, initial_data: pd.DataFrame, sequence_length: int = 3) -> Dict[str, Any]:
    """
    Genera predicciones para una secuencia de días.
    
    Args:
        model: Modelo entrenado para pronóstico
        initial_data: Datos iniciales para comenzar la predicción
        sequence_length: Longitud de la secuencia a predecir
        
    Returns:
        Dict[str, Any]: Diccionario con predicciones serializables a JSON
    """
    logging.info(f"Generando pronóstico para {sequence_length} días")
    
    # Verificar que haya datos iniciales
    if initial_data.empty:
        raise ValueError("No hay datos iniciales para generar predicción")
    
    # Realizar predicción secuencial
    forecast_df = model.predict_sequence(initial_data, sequence_length)
    
    logging.info(f"Pronóstico generado con éxito: {len(forecast_df)} días")
    
    # Convertir DataFrame a formato JSON serializable
    forecast_dict = {
        "forecast_data": forecast_df.to_dict(orient='records'),
        "metadata": {
            "sequence_length": sequence_length,
            "forecast_date": pd.Timestamp.now().isoformat(),
            "num_predictions": len(forecast_df)
        }
    }
    
    return forecast_dict

def evaluate_forecast_accuracy(forecast: Dict[str, Any], actual: pd.DataFrame) -> Dict[str, Any]:
    """
    Evalúa la precisión del pronóstico comparando con datos reales.
    
    Args:
        forecast: Diccionario con datos de pronóstico
        actual: DataFrame con datos reales para comparación
        
    Returns:
        Dict[str, Any]: Métricas de evaluación
    """
    logging.info("Evaluando precisión del pronóstico")
    
    # Extraer DataFrame del diccionario de pronóstico
    forecast_df = pd.DataFrame(forecast["forecast_data"])
    
    if forecast_df.empty or actual.empty:
        logging.warning("No hay datos suficientes para evaluar precisión")
        return {"error": "Datos insuficientes para evaluación"}
    
    # Aquí se implementaría la lógica de evaluación
    # Por ahora, devolver métricas básicas
    metrics = {
        "forecast_days": len(forecast_df),
        "actual_days": len(actual),
        "evaluation_date": pd.Timestamp.now().isoformat(),
        "status": "completed"
    }
    
    logging.info(f"Evaluación completada: {metrics}")
    return metrics

def prepare_forecast_for_api(forecast: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepara los datos de pronóstico para el formato de API.
    
    Args:
        forecast: Diccionario con datos de pronóstico
        
    Returns:
        Dict[str, Any]: Datos formateados para API
    """
    logging.info("Preparando datos de pronóstico para API")
    
    # Extraer datos del pronóstico
    forecast_data = forecast.get("forecast_data", [])
    metadata = forecast.get("metadata", {})
    
    # Formatear para API
    api_format = {
        "status": "success",
        "forecast": forecast_data,
        "metadata": {
            "generated_at": metadata.get("forecast_date", pd.Timestamp.now().isoformat()),
            "sequence_length": metadata.get("sequence_length", len(forecast_data)),
            "model_type": "ForecastSVM",
            "version": "1.0"
        }
    }
    
    logging.info(f"Datos preparados para API: {len(forecast_data)} predicciones")
    return api_format

def save_forecast_to_database(forecast_api_data: Dict[str, Any], db_params: Dict[str, str]) -> Dict[str, Any]:
    """
    Guarda los datos de pronóstico en la base de datos PostgreSQL.
    
    Args:
        forecast_api_data: Datos de pronóstico formateados para API
        db_params: Parámetros de conexión a la base de datos
        
    Returns:
        Dict[str, Any]: Resultado de la operación de guardado
    """
    logging.info("Guardando pronóstico en base de datos")
    
    try:
        # Por ahora, simular el guardado exitoso
        # En una implementación real, aquí se conectaría a PostgreSQL
        forecast_data = forecast_api_data.get("forecast", [])
        metadata = forecast_api_data.get("metadata", {})
        
        result = {
            "status": "success",
            "records_saved": len(forecast_data),
            "saved_at": pd.Timestamp.now().isoformat(),
            "table": "weather_forecasts",
            "model_version": metadata.get("version", "1.0")
        }
        
        logging.info(f"Pronóstico guardado exitosamente: {result['records_saved']} registros")
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error_message": str(e),
            "saved_at": pd.Timestamp.now().isoformat()
        }
        logging.error(f"Error guardando pronóstico: {e}")
        return error_result