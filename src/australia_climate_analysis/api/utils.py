import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from fastapi import HTTPException
from sqlalchemy import text

logger = logging.getLogger(__name__)

def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Carga un archivo JSON y maneja los errores comunes.
    
    Args:
        file_path: Ruta al archivo JSON a cargar.
        
    Returns:
        Dict: El contenido del archivo JSON.
        
    Raises:
        HTTPException: Si el archivo no se encuentra o hay un error al decodificar el JSON.
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    try:
        if not file_path.exists():
            logger.error(f"Archivo no encontrado: {file_path}")
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {file_path.name}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error al decodificar JSON: {file_path}")
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo JSON: {file_path.name}")
    except Exception as e:
        logger.error(f"Error inesperado al cargar {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error inesperado al cargar {file_path.name}")

def get_latest_versioned_file(base_dir: Union[str, Path], file_name: str) -> Path:
    """
    Obtiene la ruta al archivo más reciente en un directorio versionado.
    
    Args:
        base_dir: Directorio base donde buscar las versiones.
        file_name: Nombre del archivo a buscar en los directorios versionados.
        
    Returns:
        Path: Ruta al archivo más reciente.
        
    Raises:
        HTTPException: Si no se encuentran directorios de versión o el archivo.
    """
    base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
    
    try:
        if not base_dir.exists():
            raise FileNotFoundError(f"Directorio base no encontrado: {base_dir}")
            
        # Buscar directorios de versión (asumiendo formato YYYY-MM-DD o similar)
        version_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not version_dirs:
            raise FileNotFoundError(f"No se encontraron directorios de versión en {base_dir}")
            
        # Ordenar por nombre (asumiendo que los nombres incluyen fechas en formato ISO)
        latest_version = sorted(version_dirs)[-1]
        file_path = latest_version / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo {file_name} no encontrado en la versión más reciente {latest_version}")
            
        return file_path
    
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al obtener archivo versionado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener archivo versionado: {str(e)}")

def find_visualization_files(vis_path: Union[str, Path], extensions: List[str] = None) -> List[str]:
    """
    Lista los archivos de visualización en el directorio especificado.
    
    Args:
        vis_path: Ruta al directorio de visualizaciones.
        extensions: Lista de extensiones de archivo a incluir (por defecto: .png, .jpg, .jpeg, .svg).
        
    Returns:
        List[str]: Lista de nombres de archivo de visualizaciones.
    """
    vis_path = Path(vis_path) if isinstance(vis_path, str) else vis_path
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.svg']
    
    if not vis_path.exists():
        logger.warning(f"Directorio de visualizaciones no encontrado: {vis_path}")
        return []
    
    try:
        return [str(f.name) for f in vis_path.iterdir() 
                if f.is_file() and f.suffix.lower() in extensions]
    except Exception as e:
        logger.error(f"Error al listar visualizaciones: {str(e)}")
        return []

def get_visualization_metadata(vis_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Obtiene metadatos sobre las visualizaciones disponibles.
    
    Args:
        vis_path: Ruta al directorio de visualizaciones.
        
    Returns:
        Dict: Diccionario con metadatos de cada visualización.
    """
    vis_path = Path(vis_path) if isinstance(vis_path, str) else vis_path
    metadata = {}
    
    # Buscar archivo de metadatos (si existe)
    metadata_file = vis_path / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            logger.warning(f"Error al cargar metadatos de visualizaciones")
    
    # Si no hay metadatos, generar información básica
    vis_files = find_visualization_files(vis_path)
    for file in vis_files:
        metadata[file] = {
            "filename": file,
            "title": file.split('.')[0].replace('_', ' ').title(),
            "type": file.split('.')[-1].lower(),
            "description": "Visualización generada por el pipeline de interpretación"
        }
    
    return metadata


def format_forecast_data(forecast_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Da formato a los datos de pronóstico para la API.
    
    Args:
        forecast_data: Datos de pronóstico del modelo
        
    Returns:
        Dict: Datos formateados con estructura consistente para la API
    """
    try:
        # Verificamos que el input tenga el formato esperado
        if not all(key in forecast_data for key in ["forecast_generated_at", "forecast_period_days", "locations"]):
            # Si no tiene el formato esperado, intentamos darle formato
            formatted_data = {
                "forecast_generated_at": forecast_data.get("generated_at", "Sin fecha"),
                "forecast_period_days": forecast_data.get("sequence_length", 3),
                "locations": {}
            }
            
            # Procesamos las ubicaciones si están disponibles en otro formato
            if "data" in forecast_data and isinstance(forecast_data["data"], list):
                for item in forecast_data["data"]:
                    loc = item.get("Location")
                    if loc:
                        if loc not in formatted_data["locations"]:
                            formatted_data["locations"][loc] = []
                        formatted_data["locations"][loc].append({
                            "date": item.get("Date"),
                            "min_temp": round(float(item.get("MinTemp", 0)), 1),
                            "max_temp": round(float(item.get("MaxTemp", 0)), 1),
                            "rainfall_mm": round(float(item.get("Rainfall", 0)), 1),
                            "humidity_9am": round(float(item.get("Humidity9am", 0)), 1),
                            "humidity_3pm": round(float(item.get("Humidity3pm", 0)), 1),
                            "pressure_9am": round(float(item.get("Pressure9am", 0)), 1),
                            "pressure_3pm": round(float(item.get("Pressure3pm", 0)), 1),
                            "wind_speed_9am": round(float(item.get("WindSpeed9am", 0)), 1),
                            "wind_speed_3pm": round(float(item.get("WindSpeed3pm", 0)), 1),
                            "rain_forecast": item.get("RainTomorrow", "Unknown")
                        })
            return formatted_data
        return forecast_data
    except Exception as e:
        logger.error(f"Error al formatear datos de pronóstico: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al formatear datos de pronóstico")
        
def create_forecast_table_if_not_exists(engine) -> None:
    """
    Crea la tabla de pronósticos climáticos si no existe.
    
    Args:
        engine: Motor de conexión a la base de datos
    """
    try:
        query = """
        CREATE TABLE IF NOT EXISTS "public"."weather_forecasts" (
            id SERIAL PRIMARY KEY,
            forecast_id VARCHAR(50),
            generated_at TIMESTAMP,
            location VARCHAR(100),
            forecast_date DATE,
            min_temp FLOAT,
            max_temp FLOAT,
            rainfall_mm FLOAT,
            humidity_9am FLOAT,
            humidity_3pm FLOAT,
            pressure_9am FLOAT,
            pressure_3pm FLOAT,
            wind_speed_9am FLOAT,
            wind_speed_3pm FLOAT,
            rain_forecast VARCHAR(10),
            model_name VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        with engine.connect() as connection:
            connection.execute(text(query))
            connection.commit()
        logger.info("Tabla de pronósticos creada o verificada correctamente")
    except Exception as e:
        logger.error(f"Error al crear tabla de pronósticos: {str(e)}")
