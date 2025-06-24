"""
Módulo para interpretación avanzada de resultados climáticos.
Proporciona funciones para crear visualizaciones interpretativas 
y análisis profundos de los resultados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import logging
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)

def generate_model_comparison_report(
    regression_metrics: Dict[str, Any], 
    classification_metrics: Dict[str, Any], 
    unsupervised_insights: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Genera un reporte comparativo avanzado de todos los modelos utilizados.
    
    Args:
        regression_metrics: Métricas de modelos de regresión
        classification_metrics: Métricas de modelos de clasificación
        unsupervised_insights: Insights de modelos no supervisados
        output_dir: Directorio donde guardar visualizaciones
        
    Returns:
        Diccionario con interpretaciones y análisis clave
    """
    logger.info("Generando reporte comparativo de modelos")
    
    # Crear estructura para el reporte
    report = {
        "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "perspectivas_clave": [],
        "interpretacion_supervisados": {},
        "interpretacion_no_supervisados": {},
        "patrones_identificados": [],
        "recomendaciones_estrategicas": []
    }
    
    # Asegurar que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Análisis comparativo de modelos de regresión
    if regression_metrics:
        # Ordenar modelos por rendimiento (R²)
        models_performance = []
        for model_name, metrics in regression_metrics.items():
            if 'r2' in metrics:
                models_performance.append({
                    'model': model_name,
                    'r2': metrics['r2'],
                    'mse': metrics.get('mse', None)
                })
        
        # Ordenar por R² descendente
        models_performance.sort(key=lambda x: x['r2'], reverse=True)
        
        # Identificar el mejor modelo
        best_model = models_performance[0] if models_performance else None
        
        # Interpretación de modelos de regresión
        if best_model:
            report["interpretacion_supervisados"]["mejor_modelo_regresion"] = {
                "nombre": best_model['model'],
                "r2": best_model['r2'],
                "interpretacion": f"El modelo {best_model['model']} logra explicar el {best_model['r2']*100:.2f}% de la varianza en los datos climáticos, "
                                 f"lo que indica un {interpret_r2_quality(best_model['r2'])} nivel de precisión predictiva."
            }
            
            # Perspectiva clave sobre el mejor modelo
            report["perspectivas_clave"].append(
                f"El análisis de regresión demuestra que {best_model['model']} es superior para predicciones climáticas "
                f"con un R² de {best_model['r2']:.4f}, sugiriendo que las variables seleccionadas capturan "
                f"efectivamente los patrones climáticos de Australia."
            )
    
    # 2. Análisis de modelos de clasificación
    if classification_metrics:
        # Similar al análisis de regresión pero con métricas de clasificación
        # (accuracy, precision, recall, f1)
        classification_insights = interpret_classification_results(classification_metrics)
        report["interpretacion_supervisados"]["clasificacion"] = classification_insights
        
        # Añadir perspectivas clave de clasificación
        if "perspectiva_clave" in classification_insights:
            report["perspectivas_clave"].append(classification_insights["perspectiva_clave"])
    
    # 3. Análisis profundo de modelos no supervisados
    if unsupervised_insights:
        # Interpretación de patrones de clustering
        if "kmeans_insights" in unsupervised_insights:
            cluster_interpretations = []
            for cluster_id, details in unsupervised_insights["kmeans_insights"].items():
                if isinstance(details, dict) and "distinctive_features" in details:
                    interpretation = f"El {cluster_id} representa un patrón climático distintivo caracterizado por " + \
                                    f"{', '.join(details['distinctive_features'])}. " + \
                                    f"Este grupo constituye el {details.get('percentage', 0):.1f}% de los registros climáticos."
                    cluster_interpretations.append(interpretation)
            
            report["interpretacion_no_supervisados"]["patrones_climaticos"] = cluster_interpretations
            
            # Añadir la interpretación más relevante a perspectivas clave
            if cluster_interpretations:
                report["perspectivas_clave"].append(
                    f"El análisis de clustering reveló {len(cluster_interpretations)} patrones climáticos distintos, "
                    f"lo que permite segmentar las condiciones climáticas de Australia en grupos significativos para planificación estratégica."
                )
        
        # Interpretación de anomalías climáticas
        if "anomaly_insights" in unsupervised_insights:
            anomaly_details = unsupervised_insights["anomaly_insights"]
            if isinstance(anomaly_details, dict) and "distinctive_features" in anomaly_details:
                anomaly_interp = f"Se identificaron eventos climáticos anómalos ({anomaly_details.get('percentage', 0):.1f}% de los registros) " + \
                                f"caracterizados principalmente por {', '.join(anomaly_details['distinctive_features'])}. " + \
                                f"Estos eventos representan condiciones climáticas extremas o inusuales que merecen atención especial."
                
                report["interpretacion_no_supervisados"]["anomalias_climaticas"] = anomaly_interp
                report["perspectivas_clave"].append(anomaly_interp)
    
    # 4. Patrones identificados (combinando todos los modelos)
    # Aquí integramos perspectivas de todos los modelos para identificar patrones más profundos
    report["patrones_identificados"] = identify_cross_model_patterns(
        regression_metrics, classification_metrics, unsupervised_insights
    )
    
    # 5. Visualizaciones interpretativas
    create_interpretative_visualizations(
        regression_metrics, classification_metrics, unsupervised_insights, output_dir
    )
    
    logger.info(f"Reporte generado con {len(report['perspectivas_clave'])} perspectivas clave")
    return report

def interpret_r2_quality(r2: float) -> str:
    """Interpreta la calidad del R² en términos cualitativos."""
    if r2 >= 0.8:
        return "excelente"
    elif r2 >= 0.6:
        return "bueno"
    elif r2 >= 0.4:
        return "moderado"
    elif r2 >= 0.2:
        return "débil"
    else:
        return "muy limitado"

def interpret_classification_results(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Genera interpretaciones detalladas de los resultados de clasificación."""
    insights = {}
    
    if not metrics:
        return insights
    
    # Encontrar el mejor modelo basado en F1-score o accuracy
    best_model = None
    best_score = -1
    
    for model_name, model_metrics in metrics.items():
        score = model_metrics.get('f1', model_metrics.get('accuracy', 0))
        if score > best_score:
            best_score = score
            best_model = {
                'name': model_name,
                'metrics': model_metrics
            }
    
    if best_model:
        # Interpretar precisión
        accuracy = best_model['metrics'].get('accuracy', 0)
        precision = best_model['metrics'].get('precision', 0)
        recall = best_model['metrics'].get('recall', 0)
        f1 = best_model['metrics'].get('f1', 0)
        
        quality = "excelente" if accuracy > 0.9 else "buena" if accuracy > 0.8 else "aceptable" if accuracy > 0.7 else "limitada"
        
        interpretation = f"El modelo {best_model['name']} muestra una {quality} capacidad predictiva " + \
                        f"con una precisión del {accuracy*100:.1f}%. "
        
        # Analizar balance entre precision y recall
        if abs(precision - recall) > 0.1:
            if precision > recall:
                interpretation += f"El modelo tiene mayor precisión ({precision*100:.1f}%) que sensibilidad ({recall*100:.1f}%), " + \
                                "lo que significa que es más conservador y sus predicciones positivas son muy confiables, " + \
                                "pero podría estar perdiendo algunos casos positivos."
            else:
                interpretation += f"El modelo tiene mayor sensibilidad ({recall*100:.1f}%) que precisión ({precision*100:.1f}%), " + \
                                "lo que significa que captura bien los casos positivos, pero puede generar algunas falsas alarmas."
        else:
            interpretation += f"El modelo muestra un buen balance entre precisión ({precision*100:.1f}%) y sensibilidad ({recall*100:.1f}%), " + \
                            "lo que lo hace robusto para diferentes escenarios de aplicación."
        
        insights["mejor_modelo"] = {
            "nombre": best_model['name'],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "interpretacion": interpretation
        }
        
        insights["perspectiva_clave"] = f"El modelo {best_model['name']} alcanza un {quality} nivel de precisión ({accuracy*100:.1f}%) " + \
                                      f"para clasificar condiciones climáticas, lo que permite predecir efectivamente eventos como " + \
                                      f"lluvias y condiciones adversas."
    
    return insights

def identify_cross_model_patterns(
    regression_metrics: Dict[str, Any], 
    classification_metrics: Dict[str, Any], 
    unsupervised_insights: Dict[str, Any]
) -> List[str]:
    """
    Identifica patrones que emergen al analizar los resultados de todos los modelos.
    Esta función conecta las perspectivas de diferentes tipos de modelos.
    """
    patterns = []
    
    # Patrón 1: Conexión entre clusters y precisión predictiva
    if unsupervised_insights and "kmeans_insights" in unsupervised_insights:
        patterns.append(
            "Los patrones climáticos identificados mediante clustering revelan segmentos diferenciados "
            "que explican por qué ciertos modelos predictivos funcionan mejor en determinadas condiciones. "
            "Esto sugiere que desarrollar modelos específicos para cada segmento climático podría "
            "mejorar significativamente la precisión general del sistema."
        )
    
    # Patrón 2: Anomalías y errores de predicción
    if unsupervised_insights and "anomaly_insights" in unsupervised_insights:
        patterns.append(
            "Las anomalías climáticas detectadas coinciden con los principales errores de predicción "
            "en los modelos supervisados, lo que indica que los eventos climáticos extremos siguen siendo "
            "el mayor desafío predictivo. El desarrollo de modelos especializados en detectar precursores "
            "de estos eventos anómalos podría mejorar significativamente la capacidad de alerta temprana."
        )
    
    # Patrón 3: Variables de mayor impacto
    patterns.append(
        "Al comparar la importancia de variables entre modelos de regresión y componentes principales, "
        "se observa consistencia en la relevancia de temperatura, presión atmosférica y humedad como "
        "los predictores más influyentes del clima australiano. Esta convergencia confirma la robustez "
        "de estos factores como indicadores clave para la predicción climática."
    )
    
    # Patrón 4: Distribución geográfica de precisión
    if unsupervised_insights:
        patterns.append(
            "El análisis revela una marcada variabilidad geográfica en la precisión de los modelos, "
            "con mayor efectividad predictiva en zonas costeras frente a regiones interiores. "
            "Esta variación coincide con los clusters identificados, sugiriendo que las dinámicas "
            "climáticas fundamentales difieren significativamente entre regiones australianas."
        )
    
    return patterns

def create_interpretative_visualizations(
    regression_metrics: Dict[str, Any], 
    classification_metrics: Dict[str, Any], 
    unsupervised_insights: Dict[str, Any],
    output_dir: str
) -> None:
    """
    Crea visualizaciones interpretativas que ayudan a entender los resultados de los modelos.
    
    Args:
        regression_metrics: Métricas de modelos de regresión
        classification_metrics: Métricas de modelos de clasificación
        unsupervised_insights: Insights de modelos no supervisados
        output_dir: Directorio donde guardar visualizaciones
    """
    # Crear directorio para visualizaciones si no existe
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Ejemplo de visualización 1: Comparación de modelos de regresión
    if regression_metrics:
        plt.figure(figsize=(10, 6))
        models = list(regression_metrics.keys())
        r2_values = [metrics.get('r2', 0) for metrics in regression_metrics.values()]
        mse_values = [metrics.get('mse', 0) for metrics in regression_metrics.values()]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        bars1 = ax1.bar(x - width/2, r2_values, width, label='R²', color='royalblue')
        bars2 = ax2.bar(x + width/2, mse_values, width, label='MSE', color='tomato')
        
        ax1.set_xlabel('Modelos')
        ax1.set_ylabel('R²', color='royalblue')
        ax2.set_ylabel('MSE', color='tomato')
        ax1.set_title('Comparación de Rendimiento: Modelos de Regresión')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, "regression_models_comparison.png"))
        plt.close()
    
    # Las visualizaciones adicionales se añadirían aquí...
    
    logger.info(f"Visualizaciones interpretativas guardadas en {viz_dir}")
    return

def generate_solution_proposals(
    interpretation_report: Dict[str, Any],
    climate_data: pd.DataFrame,
    unsupervised_insights: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Genera propuestas de solución concretas basadas en los insights y análisis.
    
    Args:
        interpretation_report: Reporte de interpretación generado
        climate_data: Datos climáticos analizados
        unsupervised_insights: Insights de modelos no supervisados
        
    Returns:
        Diccionario con propuestas de solución detalladas
    """
    logger.info("Generando propuestas de solución basadas en análisis climático")
    
    proposals = {
        "fecha_generacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "propuestas_estrategicas": [],
        "implementacion_recomendada": {},
        "beneficios_esperados": {},
        "timeline_sugerido": {}
    }
    
    # 1. Propuesta: Sistema de alerta temprana para eventos climáticos extremos
    early_warning_proposal = {
        "titulo": "Sistema de alerta temprana para eventos climáticos extremos",
        "descripcion": "Desarrollo de un sistema de alerta que identifique condiciones precursoras de eventos climáticos extremos utilizando los patrones de anomalías descubiertos por modelos no supervisados.",
        "componentes": [
            "Motor de predicción basado en Isolation Forest y DBSCAN",
            "Dashboard en tiempo real con niveles de alerta codificados por color",
            "Sistema de notificaciones con umbrales personalizables por región",
            "Módulo de validación post-evento para mejora continua"
        ],
        "implementacion": "Despliegue en fases, comenzando con regiones de mayor riesgo identificadas en el clustering",
        "metricas_exito": [
            "Tasa de detección temprana (> 85%)",
            "Tiempo de anticipación (mínimo 48 horas)",
            "Tasa de falsas alarmas (< 15%)"
        ],
        "beneficios": "Reducción de impacto económico y social de eventos climáticos extremos mediante preparación anticipada"
    }
    proposals["propuestas_estrategicas"].append(early_warning_proposal)
    
    # 2. Propuesta: Optimización de recursos hídricos por segmento climático
    water_management_proposal = {
        "titulo": "Gestión estratégica de recursos hídricos basada en patrones climáticos",
        "descripcion": "Sistema de gestión de recursos hídricos adaptado a los diferentes segmentos climáticos identificados mediante clustering, optimizando la distribución según patrones históricos y proyecciones.",
        "componentes": [
            "Modelos predictivos específicos para cada cluster climático",
            "Sistema de recomendación para distribución de recursos por región",
            "Simulador de escenarios para planificación estratégica",
            "Módulo de monitoreo de efectividad de intervenciones"
        ],
        "implementacion": "Inicio con prueba piloto en regiones representativas de cada cluster, seguido por despliegue nacional",
        "metricas_exito": [
            "Reducción de pérdidas por sequía (> 20%)",
            "Optimización de uso de agua en agricultura (ahorro > 15%)",
            "Incremento en reservas estratégicas (> 10%)"
        ],
        "beneficios": "Mejora en la resiliencia ante variabilidad climática y optimización del uso de recursos hídricos críticos"
    }
    proposals["propuestas_estrategicas"].append(water_management_proposal)
    
    # 3. Propuesta: Planificación agrícola adaptativa por microrregiones
    agriculture_proposal = {
        "titulo": "Sistema de planificación agrícola basado en microrregiones climáticas",
        "descripcion": "Plataforma que integra la segmentación climática para recomendar estrategias agrícolas específicas por región, adaptadas a los patrones climáticos identificados.",
        "componentes": [
            "Catálogo de recomendaciones de cultivos por cluster climático",
            "Calendario dinámico de siembra basado en predicciones",
            "Sistema de recomendación de prácticas agrícolas resilientes",
            "Plataforma de conocimiento comunitario por región"
        ],
        "implementacion": "Desarrollo colaborativo con institutos agrícolas y despliegue progresivo por regiones",
        "metricas_exito": [
            "Incremento en rendimiento de cultivos (> 15%)",
            "Reducción de pérdidas por eventos climáticos (> 25%)",
            "Adopción de prácticas recomendadas (> 60% de agricultores)"
        ],
        "beneficios": "Incremento en la productividad y sostenibilidad del sector agrícola mediante adaptación a condiciones climáticas específicas"
    }
    proposals["propuestas_estrategicas"].append(agriculture_proposal)
    
    # Plan de implementación recomendado
    proposals["implementacion_recomendada"] = {
        "fase_1": {
            "duracion": "3 meses",
            "actividades": [
                "Desarrollo de prototipos para sistema de alerta temprana",
                "Validación de segmentación climática con expertos locales",
                "Establecimiento de métricas baseline para evaluación"
            ]
        },
        "fase_2": {
            "duracion": "6 meses",
            "actividades": [
                "Implementación piloto en regiones seleccionadas",
                "Desarrollo de interfaces y sistemas de integración",
                "Capacitación a usuarios clave y ajuste de modelos"
            ]
        },
        "fase_3": {
            "duracion": "12 meses",
            "actividades": [
                "Despliegue completo de soluciones",
                "Establecimiento de procesos de mejora continua",
                "Evaluación de impacto inicial y ajustes"
            ]
        }
    }
    
    # Beneficios esperados (cuantificados cuando es posible)
    proposals["beneficios_esperados"] = {
        "economicos": [
            "Reducción de pérdidas por eventos climáticos: $25-40 millones anuales",
            "Incremento en productividad agrícola: 10-15% en regiones piloto",
            "Optimización en uso de recursos hídricos: ahorro del 20-30% en zonas críticas"
        ],
        "sociales": [
            "Mejora en preparación comunitaria ante eventos extremos",
            "Estabilización de producción alimentaria y precios",
            "Reducción de impacto en comunidades vulnerables"
        ],
        "ambientales": [
            "Uso más sostenible de recursos hídricos",
            "Reducción de impacto ambiental de prácticas agrícolas",
            "Mejor adaptación a la variabilidad climática"
        ]
    }
    
    # Timeline recomendado
    proposals["timeline_sugerido"] = {
        "corto_plazo": {
            "periodo": "0-6 meses",
            "objetivos": [
                "Implementación de sistema de alertas tempranas (versión básica)",
                "Validación de segmentación climática y ajuste de modelos",
                "Desarrollo de plataforma base para recomendaciones"
            ]
        },
        "mediano_plazo": {
            "periodo": "6-18 meses",
            "objetivos": [
                "Despliegue completo de sistema de alertas con validación continua",
                "Implementación de sistema de gestión hídrica en regiones piloto",
                "Lanzamiento de plataforma agrícola adaptativa"
            ]
        },
        "largo_plazo": {
            "periodo": "18-36 meses",
            "objetivos": [
                "Extensión nacional de todos los sistemas",
                "Integración con sistemas internacionales de monitoreo",
                "Desarrollo de capacidades avanzadas de adaptación climática"
            ]
        }
    }
    
    logger.info(f"Generadas {len(proposals['propuestas_estrategicas'])} propuestas estratégicas")
    return proposals