import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from fastapi import HTTPException

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
