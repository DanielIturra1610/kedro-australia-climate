"""
API para el proyecto Australia Climate Analysis.

Este módulo proporciona una API REST utilizando FastAPI para acceder a los resultados
de los pipelines de análisis climático, incluyendo métricas de modelos, visualizaciones
y propuestas de solución.
"""

__version__ = "1.0.0"
__author__ = "Australia Climate Analysis Team"

# Importaciones para facilitar el uso del módulo
from .main import app
