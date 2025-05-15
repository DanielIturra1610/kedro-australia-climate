# Guía de Visualización en Power BI para Modelos de Clima en Australia

## Introducción

Este documento proporciona instrucciones para crear visualizaciones efectivas en Power BI utilizando los datos exportados de nuestros modelos de análisis de clima en Australia. Las visualizaciones están diseñadas para mostrar los resultados de los diferentes modelos de regresión (SVM, Random Forest, Regresión Múltiple) y el modelo de clasificación Naive Bayes para predicción de lluvia.

## Preparación de Datos

1. Ejecute el script de exportación para generar los archivos CSV necesarios:

```bash
python src/australia_climate_analysis/export_to_powerbi.py
```

2. Los archivos CSV se guardarán en la carpeta `data/09_powerbi/`:
   - `regression_metrics.csv`: Métricas de los modelos de regresión
   - `classification_metrics.csv`: Métricas del modelo de clasificación Naive Bayes
   - `next_day_predictions.csv`: Predicciones de lluvia para el día siguiente
   - `climate_risk_index.csv`: Índice de riesgo climático por ubicación

## Importación de Datos en Power BI

1. Abra Power BI Desktop
2. Seleccione "Obtener datos" > "Texto/CSV"
3. Navegue a la carpeta `data/09_powerbi/` y seleccione cada uno de los archivos CSV
4. Para cada archivo, verifique que los tipos de datos sean correctos:
   - `run_id`: Texto
   - `model_name`: Texto
   - `metric`: Texto
   - `value`: Decimal
   - Fechas: Fecha/Hora

## Visualizaciones Recomendadas

### 1. Dashboard de Comparación de Modelos de Regresión

**Visualización 1: Comparación de MSE por Modelo**
- Tipo: Gráfico de barras
- Datos:
  - Eje X: `model_name` (filtrado para mostrar solo modelos de regresión)
  - Eje Y: `value` (filtrado donde `metric` = "mse")
- Interpretación: Muestra qué modelo tiene el menor error cuadrático medio. Menor MSE indica mejor rendimiento.

**Visualización 2: Comparación de R² por Modelo**
- Tipo: Gráfico de barras
- Datos:
  - Eje X: `model_name` (filtrado para mostrar solo modelos de regresión)
  - Eje Y: `value` (filtrado donde `metric` = "r2")
- Interpretación: Muestra qué modelo explica mejor la varianza en los datos. Mayor R² indica mejor ajuste.

**Visualización 3: Tiempo de Predicción por Modelo**
- Tipo: Gráfico de barras
- Datos:
  - Eje X: `model_name`
  - Eje Y: `value` (filtrado donde `metric` = "prediction_time_seconds")
- Interpretación: Muestra qué modelo es más rápido en hacer predicciones. Importante para aplicaciones en tiempo real.

**Visualización 4: Tabla de Métricas Completas**
- Tipo: Tabla
- Datos: Todas las columnas de `regression_metrics.csv`
- Interpretación: Proporciona una vista detallada de todas las métricas para análisis en profundidad.

### 2. Dashboard de Rendimiento del Modelo Naive Bayes

**Visualización 1: Matriz de Confusión**
- Tipo: Mapa de calor personalizado (usando medidas DAX)
- Datos: Crear medidas para verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos basados en `precision_rain`, `recall_rain`, `precision_no_rain`, `recall_no_rain`
- Interpretación: Muestra la capacidad del modelo para clasificar correctamente los días lluviosos y no lluviosos.

**Visualización 2: Métricas de Clasificación**
- Tipo: Tarjetas múltiples
- Datos:
  - Accuracy
  - Precision (lluvia y no lluvia)
  - Recall (lluvia y no lluvia)
  - F1 Score (lluvia y no lluvia)
- Interpretación: Proporciona una vista rápida del rendimiento general del modelo.

**Visualización 3: Predicciones por Ubicación**
- Tipo: Mapa
- Datos:
  - Ubicación: `Location` de `next_day_predictions.csv`
  - Color: `RainTomorrow_Prediction` (Sí/No)
  - Tamaño: `RainTomorrow_Probability`
- Interpretación: Muestra dónde se espera que llueva mañana y con qué probabilidad.

### 3. Dashboard de Comparación de Todos los Modelos

**Visualización 1: Radar Chart de Rendimiento**
- Tipo: Gráfico de radar
- Datos:
  - Ejes: Diferentes métricas normalizadas (MSE, R², Accuracy, etc.)
  - Series: Diferentes modelos
- Interpretación: Permite comparar el rendimiento de todos los modelos en diferentes métricas en una sola vista.

**Visualización 2: Evolución del Rendimiento por Ejecución**
- Tipo: Gráfico de líneas
- Datos:
  - Eje X: `run_id`
  - Eje Y: Métricas seleccionadas
  - Series: Diferentes modelos
- Interpretación: Muestra cómo ha evolucionado el rendimiento de los modelos a lo largo de diferentes ejecuciones.

## Interpretaciones Clave para Incluir

### Modelos de Regresión

1. **SVM (Support Vector Machine)**:
   - MSE: 9.74
   - R²: 0.807
   - Tiempo de predicción: 8.34 segundos
   - Interpretación: El modelo SVM muestra un buen equilibrio entre precisión y tiempo de ejecución. La optimización realizada (cambio de kernel RBF a lineal, reducción de C de 100 a 1.0) ha mejorado significativamente el rendimiento sin sacrificar la precisión.

2. **Random Forest Regressor**:
   - MSE: 10.04
   - R²: 0.801
   - Interpretación: Ligeramente menos preciso que SVM pero ofrece buena interpretabilidad a través de la importancia de características.

3. **Regresión Múltiple**:
   - MSE: 9.73
   - R²: 0.807
   - Tiempo de predicción: 0.002 segundos
   - Interpretación: El modelo más simple es sorprendentemente preciso y extremadamente rápido, lo que lo hace ideal para aplicaciones en tiempo real.

### Modelo de Clasificación (Naive Bayes)

- Accuracy: 0.817 (81.7%)
- Precision (No lluvia): 0.884
- Recall (No lluvia): 0.884
- Precision (Lluvia): 0.572
- Recall (Lluvia): 0.571
- F1 (No lluvia): 0.884
- F1 (Lluvia): 0.571
- Tiempo de predicción: 0.008 segundos

Interpretación: El modelo Naive Bayes muestra un buen rendimiento general con un 81.7% de precisión. Es mucho mejor para predecir días sin lluvia (88.4% de precisión) que días lluviosos (57.2% de precisión), lo que refleja el desbalance en los datos de entrenamiento (77.3% días sin lluvia vs 22.7% días con lluvia). A pesar de esto, el modelo es extremadamente rápido y proporciona buenas probabilidades para la toma de decisiones.

## Conclusiones para el Dashboard

1. **Mejor Modelo para Predicción de Temperatura**: Regresión Múltiple (mejor equilibrio entre precisión y velocidad)
2. **Mejor Modelo para Interpretabilidad**: Random Forest (proporciona importancia de características)
3. **Mejor Modelo para Predicción de Lluvia**: Naive Bayes (buena precisión y extremadamente rápido)
4. **Recomendación para Implementación en Producción**: Utilizar Regresión Múltiple para predicciones numéricas y Naive Bayes para predicciones de lluvia, debido a su excelente equilibrio entre precisión y rendimiento.

## Pasos Adicionales para Mejorar las Visualizaciones

1. Añadir filtros interactivos por ubicación y fecha
2. Crear tooltips personalizados con información detallada
3. Implementar drill-down para explorar datos a nivel más granular
4. Configurar actualizaciones automáticas de datos conectando Power BI directamente a PostgreSQL
5. Crear un informe de Power BI Service para compartir con stakeholders