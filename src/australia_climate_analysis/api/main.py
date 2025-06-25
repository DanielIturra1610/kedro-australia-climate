from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import json
import logging
import os
import sqlalchemy
from sqlalchemy import create_engine, text
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date
from uuid import uuid4
from .utils import (
    load_json_file, 
    get_latest_versioned_file, 
    find_visualization_files, 
    get_visualization_metadata
)
from .utils import format_forecast_data, create_forecast_table_if_not_exists

logger = logging.getLogger(__name__)

# Modelos Pydantic para la API
class ForecastDay(BaseModel):
    date: str
    min_temp: float = None
    max_temp: float = None
    rainfall_mm: float = None
    humidity_9am: float = None
    humidity_3pm: float = None
    pressure_9am: float = None
    pressure_3pm: float = None
    wind_speed_9am: float = None
    wind_speed_3pm: float = None
    rain_forecast: str = None

class LocationForecast(BaseModel):
    location: str
    forecast_days: List[ForecastDay]

class ForecastRequest(BaseModel):
    location: Optional[str] = Field(None, description="Ubicación específica para pronóstico. Si es None, retorna todas.")
    days: int = Field(3, description="Número de días a pronosticar (máximo 3)")

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

app = FastAPI(
    title="Australia Climate Analysis API", 
    description="API para acceder a los resultados del análisis climático de Australia",
    version="1.0.0"
)

# Configurar CORS para permitir solicitudes desde el frontend
# Leer orígenes permitidos desde variable de entorno o usar valores predeterminados
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://frontend:80")
origins = [origin.strip() for origin in allowed_origins.split(",")]

logger.info(f"Configurando CORS con los siguientes orígenes permitidos: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas base del proyecto
BASE_PATH = Path(__file__).parent.parent.parent.parent
DATA_PATH = BASE_PATH / "data"
MODEL_OUTPUT_PATH = DATA_PATH / "07_model_output"
REPORTING_PATH = DATA_PATH / "08_reporting"
VISUALIZATION_PATH = REPORTING_PATH / "interpretations" / "visualizations"

# Configuración de la conexión a la base de datos
DEFAULT_DB_URL = "postgresql://kedro_user:kedro_pass@postgres:5432/climate_db"

# Funciones auxiliares para conectarse a la base de datos PostgreSQL
def create_database_engine(database_url: str) -> sqlalchemy.engine.Engine:
    """Crea un motor de base de datos a partir de una URL de conexión"""
    return create_engine(database_url)

def execute_query(engine: sqlalchemy.engine.Engine, query: str) -> List[Dict[str, Any]]:
    """Ejecuta una consulta SQL y devuelve los resultados como una lista de diccionarios"""
    with engine.connect() as connection:
        result = connection.execute(text(query))
        return [dict(row) for row in result]

def get_database_connection_string() -> Optional[str]:
    """Obtiene la cadena de conexión a la base de datos desde la variable de entorno"""
    return os.getenv("DATABASE_URL")

@app.get("/")
def read_root():
    """Endpoint principal que devuelve información sobre la API"""
    return {
        "message": "Australia Climate Analysis API", 
        "status": "online",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/api/metrics", "description": "Métricas de modelos"},
            {"path": "/api/metrics/{model_type}", "description": "Métricas filtradas por tipo de modelo"},
            {"path": "/api/runs", "description": "Lista de ejecuciones de modelos"},
            {"path": "/api/visualizations", "description": "Visualizaciones disponibles"},
            {"path": "/api/visualizations/{name}", "description": "Visualización específica"},
            {"path": "/api/forecasts/latest", "description": "Pronóstico climático más reciente"},
            {"path": "/api/forecasts/locations", "description": "Ubicaciones disponibles para pronósticos"},
            {"path": "/api/forecasts/generate", "description": "Generar nuevo pronóstico"},
            {"path": "/api/forecasts/metrics", "description": "Métricas del modelo de pronóstico"}
        ]
    }

@app.get("/api/insights/unsupervised")
def get_unsupervised_insights():
    """Obtiene los insights generados por los modelos no supervisados"""
    file_path = MODEL_OUTPUT_PATH / "unsupervised_insights.json"
    return load_json_file(file_path)

@app.get("/api/metrics/regression")
def get_regression_metrics():
    """Obtiene las métricas de los modelos de regresión"""
    file_path = REPORTING_PATH / "regression_model_metrics.json"
    return load_json_file(file_path)

@app.get("/api/metrics/classification")
def get_classification_metrics():
    """Obtiene las métricas de los modelos de clasificación"""
    file_path = REPORTING_PATH / "classification_model_metrics.json"
    return load_json_file(file_path)

@app.get("/api/metrics/model/{model_name}")
def get_model_metrics(model_name: str):
    """
    Obtiene las métricas de un modelo específico
    
    Args:
        model_name: Nombre del modelo (svm, random_forest, naive_bayes, tree, etc)
    """
    model_files = {
        "svm": REPORTING_PATH / "svm_metrics.json",
        "random_forest": REPORTING_PATH / "random_forest_metrics.json",
        "naive_bayes": REPORTING_PATH / "naive_bayes_metrics.json",
        "multiple_regression": REPORTING_PATH / "multiple_regression_metrics.json",
        "linear_regression": REPORTING_PATH / "regression_linear_metrics.json",
        "tree": REPORTING_PATH / "tree_model_metrics.json",
        "climate_risk": REPORTING_PATH / "climate_risk_metrics.json"
    }
    
    if model_name not in model_files:
        logger.warning(f"Modelo solicitado no encontrado: {model_name}")
        raise HTTPException(
            status_code=404, 
            detail=f"Modelo no encontrado. Opciones disponibles: {list(model_files.keys())}"
        )
    
    # Manejar archivos versionados específicamente
    if model_name == "tree":
        try:
            versioned_file = get_latest_versioned_file(model_files["tree"], "tree_model_metrics.json")
            return load_json_file(versioned_file)
        except HTTPException:
            raise
    else:
        return load_json_file(model_files[model_name])

@app.get("/api/visualizations")
def get_visualizations():
    """Lista todas las visualizaciones disponibles con metadatos"""
    metadata = get_visualization_metadata(VISUALIZATION_PATH)
    return {"visualizations": metadata}

@app.get("/api/visualizations/{filename}")
def get_visualization(filename: str):
    """Obtiene una visualización específica como archivo"""
    file_path = VISUALIZATION_PATH / filename
    
    if not file_path.exists():
        logger.warning(f"Archivo de visualización no encontrado: {filename}")
        raise HTTPException(status_code=404, detail="Visualización no encontrada")
    
    return FileResponse(
        str(file_path),
        media_type=f"image/{file_path.suffix.lower()[1:]}", 
        filename=filename
    )

@app.get("/api/solutions")
def get_solution_proposals():
    """Obtiene las propuestas de solución generadas por el modelo de interpretación"""
    solutions_path = REPORTING_PATH / "solution_proposals.json"
    
    if not solutions_path.exists():
        logger.warning("Archivo de propuestas de solución no encontrado")
        return {
            "message": "Las propuestas de solución no están disponibles. Ejecuta el pipeline de interpretación.",
            "available": False,
            "proposals": []
        }
    
    try:
        solutions = load_json_file(solutions_path)
        return {
            "available": True,
            "proposals": solutions
        }
    except HTTPException:
        return {
            "message": "Error al cargar propuestas de solución",
            "available": False,
            "proposals": []
        }

@app.get("/api/database/tables")
def get_database_tables():
    """Información sobre las tablas disponibles en la base de datos"""
    return {
        "tables": [
            {"name": "public.climate_risk", "description": "Tabla para el índice de riesgo climático"},
            {"name": "public.metadata.runs", "description": "Tabla para metadatos de ejecuciones"},
            {"name": "public.ml_metrics.classification", "description": "Tabla para métricas de modelos de clasificación"},
            {"name": "public.ml_metrics.regression", "description": "Tabla para métricas de modelos de regresión"}
        ]
    }

# Endpoints para consultar métricas directamente desde la base de datos
@app.get("/api/database/metrics/classification", tags=["database"])
def get_classification_metrics(run_id: Optional[str] = None, model_name: Optional[str] = None):
    """
    Obtiene las métricas de clasificación desde la base de datos.
    
    Args:
        run_id: ID opcional de ejecución para filtrar
        model_name: Nombre opcional del modelo para filtrar
    
    Returns:
        Lista de métricas de clasificación
    """
    try:
        db_url = get_database_connection_string() or DEFAULT_DB_URL
        engine = create_database_engine(db_url)
        
        query = """SELECT * FROM "public"."ml_metrics.classification" WHERE 1=1"""
        
        if run_id:
            query += f" AND run_id = '{run_id}'"
        
        if model_name:
            query += f" AND model_name = '{model_name}'"
        
        logger.info(f"Ejecutando consulta: {query}")
        result = execute_query(engine, query)
        
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error al obtener métricas de clasificación: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al consultar la base de datos: {str(e)}")

@app.get("/api/database/metrics/regression", tags=["database"])
def get_regression_metrics(run_id: Optional[str] = None, model_name: Optional[str] = None):
    """
    Obtiene las métricas de regresión desde la base de datos.
    
    Args:
        run_id: ID opcional de ejecución para filtrar
        model_name: Nombre opcional del modelo para filtrar
    
    Returns:
        Lista de métricas de regresión
    """
    try:
        db_url = get_database_connection_string() or DEFAULT_DB_URL
        engine = create_database_engine(db_url)
        
        query = """SELECT * FROM "public"."ml_metrics.regression" WHERE 1=1"""
        
        if run_id:
            query += f" AND run_id = '{run_id}'"
        
        if model_name:
            query += f" AND model_name = '{model_name}'"
            
        logger.info(f"Ejecutando consulta: {query}")
        result = execute_query(engine, query)
        
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error al obtener métricas de regresión: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al consultar la base de datos: {str(e)}")

@app.get("/api/database/metrics/model/{model_name}", tags=["database"])
def get_model_metrics(model_name: str, metric_type: Optional[str] = None):
    """
    Obtiene las métricas para un modelo específico, combinando datos de regresión y clasificación.
    
    Args:
        model_name: Nombre del modelo (ej. 'SVM', 'NaiveBayes', 'RandomForestRegressor')
        metric_type: Tipo de métrica opcional ('regression' o 'classification')
    
    Returns:
        Métricas del modelo solicitado
    """
    try:
        db_url = get_database_connection_string() or DEFAULT_DB_URL
        engine = create_database_engine(db_url)
        result = {}
        
        if metric_type is None or metric_type.lower() == 'classification':
            try:
                classification_query = f"""
                SELECT run_id, model_name, metric, value 
                FROM "public"."ml_metrics.classification" 
                WHERE model_name = '{model_name}'
                """
                classification_metrics = execute_query(engine, classification_query)
                if classification_metrics:
                    result["classification"] = classification_metrics
            except Exception as e:
                logger.warning(f"No se pudieron obtener métricas de clasificación para {model_name}: {str(e)}")
        
        if metric_type is None or metric_type.lower() == 'regression':
            try:
                regression_query = f"""
                SELECT run_id, model_name, metric, value 
                FROM "public"."ml_metrics.regression" 
                WHERE model_name = '{model_name}'
                """
                regression_metrics = execute_query(engine, regression_query)
                if regression_metrics:
                    result["regression"] = regression_metrics
            except Exception as e:
                logger.warning(f"No se pudieron obtener métricas de regresión para {model_name}: {str(e)}")
        
        if not result:
            raise HTTPException(status_code=404, detail=f"No se encontraron métricas para el modelo {model_name}")
        
        # Agregar información sobre el modelo SVM optimizado
        if model_name.lower() in ['svm', 'svr']:
            result["optimization_details"] = {
                "kernel": "linear",  # Cambio de RBF a kernel lineal
                "C": 1.0,           # Reducción del parámetro C de 100 a 1.0
                "sampling": "30% de los datos cuando el dataset es grande (>50,000 filas)",
                "dataset_size": "142,193 filas en el dataset original"
            }
            
        return {"status": "success", "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener métricas del modelo {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al consultar la base de datos: {str(e)}")

@app.get("/api/database/runs", tags=["database"])
def get_all_runs():
    """
    Obtiene la lista de todas las ejecuciones (runs) registradas en la base de datos.
    
    Returns:
        Lista de IDs de ejecuciones
    """
    try:
        db_url = get_database_connection_string() or DEFAULT_DB_URL
        engine = create_database_engine(db_url)
        
        # Consulta IDs de ejecuciones desde ambas tablas
        query = """
        SELECT DISTINCT run_id FROM (
            SELECT run_id FROM "public"."ml_metrics.regression" 
            UNION 
            SELECT run_id FROM "public"."ml_metrics.classification"
        ) AS runs
        ORDER BY run_id
        """
        
        result = execute_query(engine, query)
        runs = [row["run_id"] for row in result]
        
        return {"status": "success", "data": runs}
    except Exception as e:
        logger.error(f"Error al obtener lista de ejecuciones: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al consultar la base de datos: {str(e)}")
    
    # Endpoint para obtener el pronóstico más reciente
@app.get("/api/forecasts/latest", tags=["forecasts"], response_model=Dict[str, Any])
def get_latest_forecast(location: Optional[str] = None):
    """
    Obtiene el pronóstico climático más reciente para 3 días.
    
    Args:
        location: Ubicación opcional para filtrar el pronóstico
        
    Returns:
        Dict con los datos de pronóstico
    """
    try:
        # Intentar cargar desde el archivo JSON generado por el pipeline
        forecast_path = REPORTING_PATH / "weather_forecast_api_format.json"
        
        if not forecast_path.exists():
            # Intentar obtener desde base de datos si el archivo no existe
            return get_forecast_from_database(location)
        
        forecast_data = load_json_file(forecast_path)
        formatted_data = format_forecast_data(forecast_data)
        
        # Filtrar por ubicación si se especifica
        if location and location in formatted_data["locations"]:
            filtered_data = formatted_data.copy()
            filtered_data["locations"] = {location: formatted_data["locations"][location]}
            return {"status": "success", "data": filtered_data}
        
        return {"status": "success", "data": formatted_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener pronóstico: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener pronóstico: {str(e)}")

# Endpoint para obtener pronóstico desde base de datos
def get_forecast_from_database(location: Optional[str] = None):
    """
    Obtiene el pronóstico climático desde la base de datos.
    """
    try:
        db_url = get_database_connection_string() or DEFAULT_DB_URL
        engine = create_database_engine(db_url)
        
        # Crear la tabla si no existe
        create_forecast_table_if_not_exists(engine)
        
        # Consultar pronóstico más reciente
        base_query = """
        WITH latest_forecast AS (
            SELECT forecast_id, MAX(generated_at) as latest_gen
            FROM "public"."weather_forecasts"
            GROUP BY forecast_id
            ORDER BY latest_gen DESC
            LIMIT 1
        )
        SELECT wf.* FROM "public"."weather_forecasts" wf
        JOIN latest_forecast lf ON wf.forecast_id = lf.forecast_id
        WHERE 1=1 
        """
        
        if location:
            base_query += " AND location = :location"
            results = []
            with engine.connect() as connection:
                result = connection.execute(text(base_query), {"location": location})
                results = [dict(row) for row in result]
        else:
            base_query += " ORDER BY wf.location, wf.forecast_date"
            results = execute_query(engine, base_query)
            
        if not results:
            raise HTTPException(status_code=404, detail="No se encontraron pronósticos")
        
        # Formatear resultados para la API
        formatted_data = {
            "forecast_generated_at": results[0]["generated_at"].strftime("%Y-%m-%d %H:%M:%S"),
            "forecast_period_days": len(set(r["forecast_date"] for r in results)) if location else 3,
            "locations": {}
        }
        
        for row in results:
            loc = row["location"]
            if loc not in formatted_data["locations"]:
                formatted_data["locations"][loc] = []
                
            formatted_data["locations"][loc].append({
                "date": row["forecast_date"].strftime("%Y-%m-%d"),
                "min_temp": round(row["min_temp"], 1),
                "max_temp": round(row["max_temp"], 1),
                "rainfall_mm": round(row["rainfall_mm"], 1),
                "humidity_9am": round(row["humidity_9am"], 1),
                "humidity_3pm": round(row["humidity_3pm"], 1),
                "pressure_9am": round(row["pressure_9am"], 1),
                "pressure_3pm": round(row["pressure_3pm"], 1),
                "wind_speed_9am": round(row["wind_speed_9am"], 1),
                "wind_speed_3pm": round(row["wind_speed_3pm"], 1),
                "rain_forecast": row["rain_forecast"]
            })
        
        return {"status": "success", "data": formatted_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener pronóstico desde base de datos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener pronóstico desde base de datos: {str(e)}")

# Endpoint para generar un nuevo pronóstico bajo demanda
@app.post("/api/forecasts/generate", tags=["forecasts"])
def generate_forecast(request: ForecastRequest):
    """
    Genera un nuevo pronóstico climático bajo demanda usando los modelos entrenados.
    
    Args:
        request: Parámetros para generar el pronóstico
        
    Returns:
        Dict con el pronóstico generado
    """
    try:
        # Verificar límite de días
        if request.days > 3:
            return {"status": "error", "message": "El máximo de días a pronosticar es 3"}
        
        # Importar las dependencias necesarias de Kedro
        import sys
        import pandas as pd
        from pathlib import Path
        
        # Agregar el path del proyecto para importar los módulos de Kedro
        project_path = Path(__file__).parent.parent.parent.parent
        sys.path.append(str(project_path / "src"))
        
        from australia_climate_analysis.pipelines.weather_forecast_3days.forecast_model import ForecastModel
        from australia_climate_analysis.pipelines.weather_forecast_3days.nodes import generate_forecast as generate_forecast_node
        
        # Cargar el modelo entrenado
        model_path = project_path / "data" / "06_models" / "forecast_model_trained.pkl"
        if not model_path.exists():
            return {"status": "error", "message": "Modelo de pronóstico no encontrado. Ejecute el pipeline primero."}
        
        import pickle
        with open(model_path, 'rb') as f:
            trained_model = pickle.load(f)
        
        # Cargar datos sintéticos para usar como datos iniciales
        synthetic_data_path = project_path / "data" / "05_model_input" / "synthetic_weather_data.csv"
        if not synthetic_data_path.exists():
            return {"status": "error", "message": "Datos sintéticos no encontrados. Ejecute el pipeline primero."}
        
        synthetic_data = pd.read_csv(synthetic_data_path)
        
        # Filtrar por ubicación si se especifica
        if request.location:
            if request.location not in synthetic_data['Location'].values:
                return {"status": "error", "message": f"Ubicación '{request.location}' no encontrada"}
            initial_data = synthetic_data[synthetic_data['Location'] == request.location].head(1)
        else:
            # Usar una muestra de todas las ubicaciones
            initial_data = synthetic_data.groupby('Location').head(1)
        
        # Generar pronóstico usando el nodo de Kedro
        forecast_result = generate_forecast_node(trained_model, initial_data, request.days)
        
        # Formatear resultado para la API
        forecast_data = forecast_result.get("forecast_data", [])
        metadata = forecast_result.get("metadata", {})
        
        # Agrupar por ubicación
        locations_forecast = {}
        for prediction in forecast_data:
            location = prediction.get("Location", "Unknown")
            if location not in locations_forecast:
                locations_forecast[location] = []
            
            # Formatear predicción
            day_forecast = {
                "date": prediction.get("Date", ""),
                "min_temp": round(float(prediction.get("MinTemp", 0)), 1),
                "max_temp": round(float(prediction.get("MaxTemp", 0)), 1),
                "rainfall_mm": round(float(prediction.get("Rainfall", 0)), 1),
                "humidity_9am": round(float(prediction.get("Humidity9am", 0)), 1),
                "humidity_3pm": round(float(prediction.get("Humidity3pm", 0)), 1),
                "pressure_9am": round(float(prediction.get("Pressure9am", 0)), 1),
                "pressure_3pm": round(float(prediction.get("Pressure3pm", 0)), 1),
                "wind_speed_9am": round(float(prediction.get("WindSpeed9am", 0)), 1),
                "wind_speed_3pm": round(float(prediction.get("WindSpeed3pm", 0)), 1),
                "rain_forecast": prediction.get("RainTomorrow", "Unknown")
            }
            locations_forecast[location].append(day_forecast)
        
        return {
            "status": "success",
            "data": {
                "forecast_generated_at": metadata.get("forecast_date", datetime.now().isoformat()),
                "forecast_period_days": request.days,
                "model_type": "ForecastSVM",
                "locations": locations_forecast
            }
        }
        
    except ImportError as e:
        logger.error(f"Error importando módulos de Kedro: {str(e)}")
        return {"status": "error", "message": "Error cargando el modelo. Verifique que el pipeline se haya ejecutado correctamente."}
    except Exception as e:
        logger.error(f"Error al generar pronóstico: {str(e)}")
        return {"status": "error", "message": f"Error interno: {str(e)}"}

# Endpoint para obtener métricas específicas del modelo de pronóstico
@app.get("/api/forecasts/metrics", tags=["forecasts"])
def get_forecast_metrics():
    """
    Obtiene las métricas del modelo de pronóstico meteorológico.
    
    Returns:
        Dict con las métricas del modelo de pronóstico
    """
    try:
        # Intentar cargar métricas desde el archivo JSON generado por el pipeline
        metrics_path = REPORTING_PATH / "forecast_metrics.json"
        
        if metrics_path.exists():
            metrics_data = load_json_file(metrics_path)
            return {"status": "success", "data": metrics_data}
        
        # Si no existe el archivo, devolver métricas básicas
        return {
            "status": "info", 
            "message": "Métricas de pronóstico no disponibles. Ejecute el pipeline para generar métricas.",
            "data": {
                "model_type": "ForecastSVM",
                "status": "not_available"
            }
        }
        
    except Exception as e:
        logger.error(f"Error al obtener métricas de pronóstico: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener métricas: {str(e)}")

# Endpoint para obtener todas las ubicaciones disponibles
@app.get("/api/forecasts/locations", tags=["forecasts"])
def get_forecast_locations():
    """
    Obtiene la lista de ubicaciones disponibles para pronósticos.
    
    Returns:
        Lista de ubicaciones disponibles
    """
    try:
        # Intentar cargar desde el archivo JSON generado por el pipeline
        forecast_path = REPORTING_PATH / "weather_forecast_api_format.json"
        
        if forecast_path.exists():
            forecast_data = load_json_file(forecast_path)
            formatted_data = format_forecast_data(forecast_data)
            locations = list(formatted_data["locations"].keys())
            return {"status": "success", "data": locations}
        
        # Intentar obtener desde base de datos si el archivo no existe
        db_url = get_database_connection_string() or DEFAULT_DB_URL
        engine = create_database_engine(db_url)
        query = """
        SELECT DISTINCT location FROM "public"."weather_forecasts"
        ORDER BY location
        """
        results = execute_query(engine, query)
        locations = [row["location"] for row in results]
        
        if not locations:
            # Si no hay datos en la base de datos, cargar desde el archivo raw
            # Esta es una opción de fallback
            weather_raw_path = DATA_PATH / "01_raw" / "weatherAUS.csv"
            
            if weather_raw_path.exists():
                import pandas as pd
                weather_raw = pd.read_csv(weather_raw_path)
                locations = sorted(weather_raw["Location"].unique().tolist())
            
        return {"status": "success", "data": locations}
    except Exception as e:
        logger.error(f"Error al obtener ubicaciones: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener ubicaciones: {str(e)}")

# Endpoint para guardar pronóstico en la base de datos
@app.post("/api/forecasts/save", tags=["forecasts"])
def save_forecast_to_db(forecast_data: Dict[str, Any]):
    """
    Guarda un pronóstico en la base de datos.
    Este endpoint es para uso interno del pipeline y no debería exponerse públicamente.
    """
    try:
        db_url = get_database_connection_string() or DEFAULT_DB_URL
        engine = create_database_engine(db_url)
        
        # Crear la tabla si no existe
        create_forecast_table_if_not_exists(engine)
        
        # Formatear datos
        formatted_data = format_forecast_data(forecast_data)
        forecast_id = str(uuid4())
        generated_at = datetime.now()
        
        # Preparar registros para insertar
        records = []
        for location, forecasts in formatted_data["locations"].items():
            for forecast in forecasts:
                record = {
                    "forecast_id": forecast_id,
                    "generated_at": generated_at,
                    "location": location,
                    "forecast_date": forecast.get("date"),
                    "min_temp": forecast.get("min_temp"),
                    "max_temp": forecast.get("max_temp"),
                    "rainfall_mm": forecast.get("rainfall_mm"),
                    "humidity_9am": forecast.get("humidity_9am"),
                    "humidity_3pm": forecast.get("humidity_3pm"),
                    "pressure_9am": forecast.get("pressure_9am"),
                    "pressure_3pm": forecast.get("pressure_3pm"),
                    "wind_speed_9am": forecast.get("wind_speed_9am"),
                    "wind_speed_3pm": forecast.get("wind_speed_3pm"),
                    "rain_forecast": forecast.get("rain_forecast"),
                    "model_name": "SequentialForecastSVM"
                }
                records.append(record)
        
        # Insertar en la base de datos
        with engine.begin() as conn:
            for record in records:
                insert_query = """
                INSERT INTO "public"."weather_forecasts" (
                    forecast_id, generated_at, location, forecast_date, min_temp, max_temp,
                    rainfall_mm, humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
                    wind_speed_9am, wind_speed_3pm, rain_forecast, model_name
                ) VALUES (
                    :forecast_id, :generated_at, :location, :forecast_date, :min_temp, :max_temp,
                    :rainfall_mm, :humidity_9am, :humidity_3pm, :pressure_9am, :pressure_3pm,
                    :wind_speed_9am, :wind_speed_3pm, :rain_forecast, :model_name
                )
                """
                conn.execute(text(insert_query), record)
        
        return {"status": "success", "forecast_id": forecast_id, "records_saved": len(records)}
    except Exception as e:
        logger.error(f"Error al guardar pronóstico: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al guardar pronóstico: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando Australia Climate Analysis API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
