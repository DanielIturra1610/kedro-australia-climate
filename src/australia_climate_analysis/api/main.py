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

from .utils import (
    load_json_file, 
    get_latest_versioned_file, 
    find_visualization_files, 
    get_visualization_metadata
)

logger = logging.getLogger(__name__)

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
            {"path": "/api/insights/unsupervised", "description": "Insights de modelos no supervisados"},
            {"path": "/api/metrics/regression", "description": "Métricas de modelos de regresión"},
            {"path": "/api/metrics/classification", "description": "Métricas de modelos de clasificación"},
            {"path": "/api/metrics/model/{model_name}", "description": "Métricas de un modelo específico"},
            {"path": "/api/visualizations", "description": "Lista de visualizaciones disponibles"},
            {"path": "/api/visualizations/{filename}", "description": "Obtiene una visualización específica"},
            {"path": "/api/solutions", "description": "Propuestas de solución basadas en el análisis"},
            {"path": "/api/database/tables", "description": "Información sobre las tablas disponibles en la base de datos"},
            {"path": "/api/database/metrics/classification", "description": "Métricas de clasificación desde la base de datos"},
            {"path": "/api/database/metrics/regression", "description": "Métricas de regresión desde la base de datos"},
            {"path": "/api/database/metrics/model/{model_name}", "description": "Métricas de un modelo específico desde la base de datos"},
            {"path": "/api/database/runs", "description": "Lista de ejecuciones registradas en la base de datos"}
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

if __name__ == "__main__":
    import uvicorn
    logger.info("Iniciando Australia Climate Analysis API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
