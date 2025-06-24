"""
Module for collecting and combining metrics from different models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def combine_regression_metrics(
    svm_metrics: Dict[str, Any] = None,
    random_forest_metrics: Dict[str, Any] = None,
    multiple_regression_metrics: Dict[str, Any] = None,
    regression_linear_metrics: Dict[str, Any] = None,
    regression_tree_metrics: Dict[str, Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Combina las métricas de todos los modelos de regresión en un solo diccionario.
    
    Args:
        svm_metrics: Métricas del modelo SVM
        random_forest_metrics: Métricas del modelo Random Forest
        multiple_regression_metrics: Métricas del modelo de regresión múltiple
        regression_linear_metrics: Métricas del modelo de regresión lineal
        regression_tree_metrics: Métricas del modelo de árbol de regresión
        
    Returns:
        Dict con todas las métricas combinadas, organizado por modelo
    """
    combined_metrics = {}
    
    # Recopilar métricas de cada modelo si están disponibles
    if svm_metrics:
        combined_metrics["SVM"] = extract_key_regression_metrics(svm_metrics, "SVM")
    
    if random_forest_metrics:
        combined_metrics["RandomForest"] = extract_key_regression_metrics(random_forest_metrics, "RandomForest")
    
    if multiple_regression_metrics:
        combined_metrics["MultipleRegression"] = extract_key_regression_metrics(multiple_regression_metrics, "MultipleRegression")
    
    if regression_linear_metrics:
        combined_metrics["LinearRegression"] = extract_key_regression_metrics(regression_linear_metrics, "LinearRegression")
    
    if regression_tree_metrics:
        combined_metrics["RegressionTree"] = extract_key_regression_metrics(regression_tree_metrics, "RegressionTree")
    
    # Añadir metadatos
    combined_metrics["_meta"] = {
        "count": len(combined_metrics),
        "models": list(combined_metrics.keys())
    }
    
    logger.info(f"Combinadas métricas de {len(combined_metrics) - 1} modelos de regresión")
    return combined_metrics

def combine_classification_metrics(
    naive_bayes_metrics: Dict[str, Any] = None,
    climate_risk_metrics: Dict[str, Any] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Combina las métricas de todos los modelos de clasificación en un solo diccionario.
    
    Args:
        naive_bayes_metrics: Métricas del modelo Naive Bayes
        climate_risk_metrics: Métricas del modelo de riesgo climático
        
    Returns:
        Dict con todas las métricas combinadas, organizado por modelo
    """
    combined_metrics = {}
    
    # Recopilar métricas de cada modelo si están disponibles
    if naive_bayes_metrics:
        combined_metrics["NaiveBayes"] = extract_key_classification_metrics(naive_bayes_metrics, "NaiveBayes")
    
    if climate_risk_metrics:
        combined_metrics["ClimateRisk"] = extract_key_classification_metrics(climate_risk_metrics, "ClimateRisk")
    
    # Añadir metadatos
    combined_metrics["_meta"] = {
        "count": len(combined_metrics),
        "models": list(combined_metrics.keys())
    }
    
    logger.info(f"Combinadas métricas de {len(combined_metrics) - 1} modelos de clasificación")
    return combined_metrics

def extract_key_regression_metrics(metrics: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Extrae y estandariza las métricas clave para modelos de regresión.
    
    Args:
        metrics: Métricas originales del modelo
        model_name: Nombre del modelo
        
    Returns:
        Dict con métricas estandarizadas
    """
    standardized_metrics = {
        "model_name": model_name,
        "type": "regression"
    }
    
    # Métricas comunes para modelos de regresión
    key_metrics = ["r2", "mse", "rmse", "mae", "mape", "training_time", "prediction_time"]
    
    for metric in key_metrics:
        # Buscar la métrica con diferentes convenciones de nomenclatura
        for potential_key in [metric, metric.upper(), f"{metric}_score", f"{model_name}_{metric}"]:
            if potential_key in metrics:
                standardized_metrics[metric] = metrics[potential_key]
                break
    
    # Para coeficientes o feature importance (si existen)
    for feature_key in ["coefficients", "feature_importance", "coef", "feature_importances"]:
        if feature_key in metrics:
            standardized_metrics["feature_importance"] = metrics[feature_key]
            break
    
    return standardized_metrics

def extract_key_classification_metrics(metrics: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Extrae y estandariza las métricas clave para modelos de clasificación.
    
    Args:
        metrics: Métricas originales del modelo
        model_name: Nombre del modelo
        
    Returns:
        Dict con métricas estandarizadas
    """
    standardized_metrics = {
        "model_name": model_name,
        "type": "classification"
    }
    
    # Métricas comunes para modelos de clasificación
    key_metrics = ["accuracy", "precision", "recall", "f1", "auc", "training_time", "prediction_time"]
    
    for metric in key_metrics:
        # Buscar la métrica con diferentes convenciones de nomenclatura
        for potential_key in [metric, metric.upper(), f"{metric}_score", f"{model_name}_{metric}"]:
            if potential_key in metrics:
                standardized_metrics[metric] = metrics[potential_key]
                break
    
    # Para matrices de confusión (si existen)
    for cm_key in ["confusion_matrix", "cm", "conf_matrix"]:
        if cm_key in metrics:
            standardized_metrics["confusion_matrix"] = metrics[cm_key]
            break
    
    return standardized_metrics

def generate_model_performance_summary(
    regression_metrics: Dict[str, Dict[str, Any]],
    classification_metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Genera un resumen comparativo del rendimiento de todos los modelos.
    
    Args:
        regression_metrics: Métricas combinadas de modelos de regresión
        classification_metrics: Métricas combinadas de modelos de clasificación
        
    Returns:
        Dict con resumen de rendimiento y comparaciones
    """
    summary = {
        "total_models": (len(regression_metrics) - 1) + (len(classification_metrics) - 1),
        "regression_models": {
            "count": len(regression_metrics) - 1,
            "models": regression_metrics.get("_meta", {}).get("models", []),
            "best_model": None,
            "comparative_metrics": {}
        },
        "classification_models": {
            "count": len(classification_metrics) - 1,
            "models": classification_metrics.get("_meta", {}).get("models", []),
            "best_model": None,
            "comparative_metrics": {}
        }
    }
    
    # Análisis de modelos de regresión
    if len(regression_metrics) > 1:  # Si hay al menos un modelo (excluyendo _meta)
        # Comparar R²
        r2_values = {model: metrics.get("r2", 0) 
                    for model, metrics in regression_metrics.items() 
                    if model != "_meta"}
        
        if r2_values:
            best_r2_model = max(r2_values, key=r2_values.get)
            summary["regression_models"]["best_model"] = {
                "name": best_r2_model,
                "r2": r2_values[best_r2_model],
                "reason": "Highest R² Score"
            }
            summary["regression_models"]["comparative_metrics"]["r2"] = r2_values
        
        # Comparar MSE
        mse_values = {model: metrics.get("mse", float('inf')) 
                     for model, metrics in regression_metrics.items() 
                     if model != "_meta"}
        
        if mse_values:
            summary["regression_models"]["comparative_metrics"]["mse"] = mse_values
    
    # Análisis de modelos de clasificación
    if len(classification_metrics) > 1:  # Si hay al menos un modelo (excluyendo _meta)
        # Comparar F1
        f1_values = {model: metrics.get("f1", 0) 
                    for model, metrics in classification_metrics.items() 
                    if model != "_meta"}
        
        if f1_values:
            best_f1_model = max(f1_values, key=f1_values.get)
            summary["classification_models"]["best_model"] = {
                "name": best_f1_model,
                "f1": f1_values[best_f1_model],
                "reason": "Highest F1 Score"
            }
            summary["classification_models"]["comparative_metrics"]["f1"] = f1_values
        
        # Comparar precision
        precision_values = {model: metrics.get("precision", 0) 
                           for model, metrics in classification_metrics.items() 
                           if model != "_meta"}
        
        if precision_values:
            summary["classification_models"]["comparative_metrics"]["precision"] = precision_values
    
    logger.info(f"Generado resumen comparativo de {summary['total_models']} modelos")
    return summary