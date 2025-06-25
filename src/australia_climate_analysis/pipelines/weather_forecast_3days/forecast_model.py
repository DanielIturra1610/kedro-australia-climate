from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List, Tuple
import logging

class ForecastModel:
    def __init__(self, kernel='linear', C=1.0, sample_threshold=50000, sample_ratio=0.3):
        self.kernel = kernel
        self.C = C
        self.sample_threshold = sample_threshold
        self.sample_ratio = sample_ratio
        self.models = {}
        self.scalers = {}
        self.metrics = {"training_time": {}, "model_size": {}, "prediction_time": {}}
    
    def _sample_data(self, X, y):
        """
        Muestrear datos si el dataset es demasiado grande para mejorar rendimiento.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            X_sampled, y_sampled: Datos muestreados si es necesario
        """
        if len(X) > self.sample_threshold:
            logging.info(f"  Muestreando datos ({self.sample_ratio*100:.0f}% de {len(X)} filas)")
            n_samples = int(len(X) * self.sample_ratio)
            indices = np.random.choice(len(X), n_samples, replace=False)
            return X.iloc[indices], y.iloc[indices]
        return X, y
    
    def train_model(self, X_train, y_train, variable, day=1):
        """
        Entrena un modelo para una variable específica y un día específico.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Serie con valores objetivo
            variable: Nombre de la variable a predecir
            day: Día para el que se entrena el modelo (1, 2, o 3)
        """
        model_key = f"{variable.lower()}_day_{day}"
        logging.info(f"Entrenando modelo para {model_key}...")
        
        # Asegurarnos de que X_train y y_train tengan índices reset para alinearlos
        X_reset = X_train.reset_index(drop=True)
        y_reset = y_train.reset_index(drop=True)
        
        # Eliminar filas con valores NaN en y
        valid_mask = ~y_reset.isna()
        if valid_mask.sum() == 0:
            logging.warning(f"No hay datos válidos para entrenar {model_key}. Omitiendo.")
            return
            
        X_valid = X_reset[valid_mask]
        y_valid = y_reset[valid_mask]
        
        # Eliminar filas con valores NaN en X
        nan_mask = X_valid.isna().any(axis=1)
        if nan_mask.sum() > 0:
            logging.info(f"Eliminando {nan_mask.sum()} filas con valores NaN en features para {model_key}")
            X_valid = X_valid[~nan_mask]
            y_valid = y_valid[~nan_mask]
        
        if len(X_valid) == 0:
            logging.warning(f"No hay datos válidos para entrenar {model_key} después de eliminar NaNs. Omitiendo.")
            return
        
        # Muestrear datos si es necesario
        X_sampled, y_sampled = self._sample_data(X_valid, y_valid)
        
        # Crear y entrenar scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sampled)
        
        # Verificación adicional para NaN después del escalado
        if np.isnan(X_scaled).any():
            logging.warning(f"Detectados NaNs después del escalado para {model_key}. Imputando con ceros.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Crear y entrenar modelo
        start_time = time.time()
        model = SVR(kernel=self.kernel, C=self.C)
        model.fit(X_scaled, y_sampled.values.ravel())
        training_time = time.time() - start_time
        
        # Guardar modelo y scaler
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        
        # Guardar métricas
        self.metrics["training_time"][model_key] = training_time
        self.metrics["model_size"][model_key] = model.__sizeof__()
        
        logging.info(f"  Modelo para {model_key} entrenado en {training_time:.2f} segundos")
    
    def train_rain_model(self, X_train, y_train, day=1):
        """
        Entrena un modelo de clasificación específico para la predicción de lluvia.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Serie con valores objetivo (0/1 o True/False)
            day: Día para el que se entrena el modelo (1, 2, o 3)
        """
        model_key = f"rain_day_{day}"
        logging.info(f"Entrenando modelo de lluvia para día {day}...")
        
        # Asegurarnos de que X_train y y_train tengan índices reset para alinearlos
        X_reset = X_train.reset_index(drop=True)
        y_reset = y_train.reset_index(drop=True)
        
        # Eliminar filas con valores NaN en y
        valid_mask = ~y_reset.isna()
        if valid_mask.sum() == 0:
            logging.warning(f"No hay datos válidos para entrenar {model_key}. Omitiendo.")
            return
            
        X_valid = X_reset[valid_mask]
        y_valid = y_reset[valid_mask]
        
        # Convertir y_valid a enteros si no lo está ya (manejar booleanos)
        if y_valid.dtype == 'bool':
            y_valid = y_valid.astype(int)
        elif y_valid.dtype == 'Int64':  # Pandas nullable integer
            y_valid = y_valid.astype(int)
        
        # Verificar que tenemos al menos dos clases para clasificación
        unique_classes = y_valid.unique()
        if len(unique_classes) < 2:
            logging.warning(f"Solo se encontró una clase ({unique_classes}) para {model_key}. Se necesitan al menos 2 clases para clasificación. Omitiendo.")
            return
        
        # Eliminar filas con valores NaN en X
        nan_mask = X_valid.isna().any(axis=1)
        if nan_mask.sum() > 0:
            logging.info(f"Eliminando {nan_mask.sum()} filas con valores NaN en features para {model_key}")
            X_valid = X_valid[~nan_mask]
            y_valid = y_valid[~nan_mask]
        
        if len(X_valid) == 0:
            logging.warning(f"No hay datos válidos para entrenar {model_key} después de eliminar NaNs. Omitiendo.")
            return
        
        # Muestrear datos si es necesario
        X_sampled, y_sampled = self._sample_data(X_valid, y_valid)
        
        # Crear y entrenar scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sampled)
        
        # Verificación adicional para NaN después del escalado
        if np.isnan(X_scaled).any():
            logging.warning(f"Detectados NaNs después del escalado para {model_key}. Imputando con ceros.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            
        # Crear y entrenar modelo
        start_time = time.time()
        model = SVC(kernel=self.kernel, C=self.C, probability=True)
        
        # Asegurar que y_sampled sea un array de enteros
        y_train_final = y_sampled.values.ravel().astype(int)
        
        model.fit(X_scaled, y_train_final)
        training_time = time.time() - start_time
        
        # Guardar modelo y scaler
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        
        # Guardar métricas
        self.metrics["training_time"][model_key] = training_time
        self.metrics["model_size"][model_key] = model.__sizeof__()
        
        logging.info(f"  Modelo para {model_key} entrenado en {training_time:.2f} segundos con {len(unique_classes)} clases: {sorted(unique_classes)}")
    
    def predict(self, X, target_vars=None, days=[1, 2, 3]):
        """
        Realiza predicciones para múltiples variables objetivo y días.
        
        Args:
            X: Features para predicción
            target_vars: Lista de variables objetivo a predecir (opcional)
            days: Lista de días para los que predecir (default: [1, 2, 3])
            
        Returns:
            DataFrame con las predicciones
        """
        # Preparar DataFrame para resultados
        results = pd.DataFrame(index=X.index)
        
        # Filtrar variables si se especifican
        vars_to_predict = target_vars if target_vars else ["MaxTemp", "MinTemp", "Rainfall", "Humidity", "rain"]
        
        # Para cada combinación de variable y día, predecir si el modelo existe
        start_time = time.time()
        for day in days:
            for var in vars_to_predict:
                # Manejar modelos rain_day_X separadamente
                if var.lower() == "rain":
                    model_key = f"rain_day_{day}"
                else:
                    model_key = f"{var.lower()}_day_{day}"
                    
                # Verificar si existe el modelo
                if model_key in self.models and model_key in self.scalers:
                    # Manejar valores NaN en X para predicción
                    X_pred = X.copy()
                    
                    # Eliminar filas con valores NaN
                    nan_mask = X_pred.isna().any(axis=1)
                    if nan_mask.sum() > 0:
                        logging.warning(f"Hay {nan_mask.sum()} filas con NaN en la predicción para {model_key}")
                        
                    # Asegurarnos de tener las mismas columnas que se usaron para entrenar
                    X_scaled = self.scalers[model_key].transform(X_pred.fillna(0))  # Llenar NaN con 0 para predicción
                    
                    # Predecir
                    if "rain" in model_key:
                        # Predicción probabilística para lluvia (probabilidad de clase 1)
                        probs = self.models[model_key].predict_proba(X_scaled)
                        results[model_key] = probs[:, 1]  # Probabilidad de clase positiva
                    else:
                        # Predicción de regresión para otras variables
                        preds = self.models[model_key].predict(X_scaled)
                        results[model_key] = preds
                else:
                    logging.warning(f"Modelo {model_key} no encontrado. Omitiendo predicción.")
        
        prediction_time = time.time() - start_time
        logging.info(f"Predicciones generadas en {prediction_time:.2f} segundos")
        
        # Guardar métrica de tiempo
        self.metrics["prediction_time"]["all"] = prediction_time
        
        return results
    
    def predict_sequence(self, initial_data: pd.DataFrame, sequence_length: int = 3) -> pd.DataFrame:
        """
        Genera predicciones secuenciales para múltiples días.
        
        Args:
            initial_data: DataFrame con datos iniciales para comenzar la predicción
            sequence_length: Número de días a predecir
            
        Returns:
            pd.DataFrame: DataFrame con predicciones para cada día
        """
        logging.info(f"Iniciando predicción secuencial para {sequence_length} días")
        logging.info(f"Datos iniciales: {initial_data.shape} filas, columnas: {list(initial_data.columns)}")
        
        # Validar datos iniciales
        if initial_data.empty:
            raise ValueError("Los datos iniciales están vacíos")
        
        # Procesar los datos iniciales de la misma manera que en el entrenamiento
        current_data = initial_data.iloc[[0]].copy()
        
        # Convertir Date a tipo datetime si existe y no es datetime ya
        if 'Date' in current_data.columns:
            if not pd.api.types.is_datetime64_dtype(current_data["Date"]):
                try:
                    current_data["Date"] = pd.to_datetime(current_data["Date"])
                    logging.info("Columna Date convertida a datetime")
                except Exception as e:
                    logging.warning(f"No se pudo convertir Date a datetime: {e}")
        else:
            logging.info("No hay columna Date en los datos iniciales")
        
        # Agregar características temporales si no existen
        if 'Date' in current_data.columns:
            if 'Month' not in current_data.columns:
                current_data["Month"] = current_data["Date"].dt.month
            if 'DayOfYear' not in current_data.columns:
                current_data["DayOfYear"] = current_data["Date"].dt.dayofyear
        else:
            # Si no hay Date, usar valores por defecto o extraer de otra fuente
            if 'Month' not in current_data.columns:
                current_data["Month"] = 6  # Valor por defecto (junio)
                logging.info("Agregada columna Month con valor por defecto")
            if 'DayOfYear' not in current_data.columns:
                current_data["DayOfYear"] = 150  # Valor por defecto (día del año)
                logging.info("Agregada columna DayOfYear con valor por defecto")
        
        # Crear variables adicionales
        if 'MaxMinTempRange' not in current_data.columns and 'MaxTemp' in current_data.columns and 'MinTemp' in current_data.columns:
            current_data["MaxMinTempRange"] = current_data["MaxTemp"] - current_data["MinTemp"]
        
        # Mapear Location a una codificación numérica si existe
        if 'Location' in current_data.columns and 'LocationCode' not in current_data.columns:
            # Usar un mapeo simple basado en las principales ciudades australianas
            location_mapping = {
                'Sydney': 0, 'Melbourne': 1, 'Brisbane': 2, 'Perth': 3, 
                'Adelaide': 4, 'Hobart': 5, 'Darwin': 6, 'Canberra': 7
            }
            current_data["LocationCode"] = current_data["Location"].map(location_mapping).fillna(0)
        
        # Filtrar características para que coincidan con las usadas en entrenamiento
        available_features = ["Month", "DayOfYear", "MinTemp", "MaxTemp", "Rainfall",
                              "Evaporation", "Sunshine", "WindGustSpeed", "WindSpeed9am",
                              "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", 
                              "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
                              "MaxMinTempRange", "LocationCode"]
        
        # Agregar características rain_day_X que esperan los modelos
        # Estas se usan como features de entrada, no como targets
        for day in [1, 2, 3]:
            col_name = f"rain_day_{day}"
            if col_name not in current_data.columns:
                # Usar un valor por defecto basado en RainToday si existe
                if 'RainToday' in current_data.columns:
                    current_data[col_name] = (current_data['RainToday'] == 'Yes').astype(int)
                else:
                    current_data[col_name] = 0  # Valor por defecto: no lluvia
            available_features.append(col_name)
        
        # Filtrar solo las características disponibles en los datos
        available_features = [f for f in available_features if f in current_data.columns]
        logging.info(f"Características disponibles para predicción: {available_features}")
        
        # Extraer características
        X_base = current_data[available_features].copy()
        
        # Rellenar valores faltantes
        X_base = X_base.fillna(0)
        
        # Lista para almacenar todas las predicciones
        all_predictions = []
        
        # Variables que podemos predecir
        predictable_vars = ["MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm", 
                          "Pressure9am", "Pressure3pm", "WindSpeed9am", "WindSpeed3pm", "Rain"]
        
        for day in range(1, sequence_length + 1):
            day_predictions = {"Day": day}
            
            # Predecir cada variable para este día
            for var in predictable_vars:
                if var.lower() == "rain":
                    model_key = f"rain_day_{min(day, 3)}"  # Usar modelo del día 3 para días posteriores
                else:
                    model_key = f"{var.lower()}_day_{min(day, 3)}"  # Usar modelo del día 3 para días posteriores
                
                if model_key in self.models and model_key in self.scalers:
                    try:
                        # Preparar datos para predicción
                        X_pred = X_base.copy()
                        
                        # Escalar los datos usando el scaler específico del modelo
                        X_scaled = self.scalers[model_key].transform(X_pred)
                        
                        # Hacer predicción
                        if "rain" in model_key:
                            # Predicción probabilística para lluvia
                            if hasattr(self.models[model_key], 'predict_proba'):
                                probs = self.models[model_key].predict_proba(X_scaled)
                                prediction = probs[0, 1]  # Probabilidad de lluvia
                            else:
                                prediction = self.models[model_key].predict(X_scaled)[0]
                            day_predictions[f"{var}_probability"] = prediction
                            day_predictions[f"{var}_prediction"] = "Yes" if prediction > 0.5 else "No"
                        else:
                            # Predicción de regresión
                            prediction = self.models[model_key].predict(X_scaled)[0]
                            day_predictions[f"{var}_prediction"] = prediction
                            
                    except Exception as e:
                        logging.warning(f"Error al predecir {var} para día {day}: {str(e)}")
                        day_predictions[f"{var}_prediction"] = None
                else:
                    logging.warning(f"Modelo {model_key} no disponible para predicción")
                    day_predictions[f"{var}_prediction"] = None
            
            all_predictions.append(day_predictions)
            
            # Para predicciones futuras, podríamos usar las predicciones actuales
            # como input para el siguiente día (esto es simplificado)
            
        # Convertir a DataFrame
        forecast_df = pd.DataFrame(all_predictions)
        
        logging.info(f"Secuencia de predicciones generada exitosamente: {len(forecast_df)} días")
        return forecast_df
    
    def get_metrics(self):
        """
        Devuelve las métricas de rendimiento del modelo.
        """
        return self.metrics