from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    imputar_min_max_temp,
    imputar_rain_today,
    convertir_fecha,
    extraer_caracteristicas_temporales,
    detectar_outliers_rainfall,
    train_climate_risk_classifier,
    predecir_riesgo_climatico
)
from .inferencia_modelo import inferir_riesgo_climatico
from .nodes import preparar_climate_inference_input
from australia_climate_analysis.pipelines.climate_risk.nodes import infer_climate_risk


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        # ────────────────────────
        # PREPROCESAMIENTO
        # ────────────────────────
        node(
            func=imputar_min_max_temp,
            inputs="weather_raw",
            outputs="weather_imputed_temp",
            name="imputar_min_max_temp_node"
        ),
        node(
            func=imputar_rain_today,
            inputs="weather_imputed_temp",
            outputs="weather_imputed_rain_today",
            name="imputar_rain_today_node"
        ),
        node(
            func=convertir_fecha,
            inputs="weather_imputed_rain_today",
            outputs="weather_date_converted",
            name="convertir_fecha_node"
        ),
        node(
            func=extraer_caracteristicas_temporales,
            inputs="weather_date_converted",
            outputs="weather_temporal_features",
            name="extraer_caracteristicas_temporales_node"
        ),
        node(
            func=detectar_outliers_rainfall,
            inputs="weather_temporal_features",
            outputs="weather_with_rainfall_outliers",
            name="detectar_outliers_rainfall_node"
        ),

        # ────────────────────────
        # ENTRENAMIENTO
        # ────────────────────────
        node(
            func=train_climate_risk_classifier,
            inputs="weather_with_rainfall_outliers",
            outputs=["modelo_clasificacion", "climate_risk_predictions"],
            name="train_climate_risk_classifier_node"
        ),

        # ────────────────────────
        # INFERENCIA
        # ────────────────────────
        node(
            func=preparar_climate_inference_input,
            inputs="weather_with_rainfall_outliers",
            outputs="climate_inference_input",
            name="preparar_climate_inference_input_node"
        ),
        node(
            func=infer_climate_risk,
            inputs=["modelo_clasificacion", "climate_inference_input"],
            outputs="predicciones_climaticas",
            name="inferir_riesgo_climatico_node"
        )
    ])


def create_inference_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=infer_climate_risk,
            inputs=["modelo_clasificacion", "climate_inference_input"],
            outputs="climate_inference_output",
            name="inferir_riesgo_climatico_node"
        )
    ])
