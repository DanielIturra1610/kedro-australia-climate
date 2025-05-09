from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    imputar_min_max_temp,
    imputar_rain_today,
    convertir_fecha,
    extraer_caracteristicas_temporales,
    detectar_outliers_rainfall,
    train_climate_risk_classifier,
    evaluate_climate_classifier,
)
from .nodes import calcular_indice_riesgo_climatico
from .inferencia_modelo import inferir_riesgo_climatico
from .nodes import preparar_climate_inference_input
from australia_climate_analysis.pipelines.climate_risk.nodes import infer_climate_risk
from australia_climate_analysis.utils.run_id import new_run

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # ────────────────────────
        # PREPROCESAMIENTO
        # ────────────────────────
        node(
            func=new_run,
            inputs=None,
            outputs=["run_id", "metadata_runs"],
            name="register_run_node",
        ),
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
            outputs=["modelo_clasificacion", "X_test_cls", "y_test_cls"],
            name="train_climate_risk_classifier_node",
        ),
        node(
            func=evaluate_climate_classifier,
            inputs=["modelo_clasificacion", "X_test_cls", "y_test_cls", "run_id"],
            outputs=["classification_metrics_local",   # puedes usar Memory o JSON si quieres
                    "classification_metrics_pg"],     # ➜ va a PostgreSQL
            name="evaluate_climate_classifier_node",
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
            outputs="predicciones_climaticas",          # <— así coincide
            name="inferir_riesgo_climatico_node"
        ),
        node(
            func=calcular_indice_riesgo_climatico,
            inputs=["predicciones_climaticas", "run_id"],   # ← ahora recibe run_id
            outputs="risk_index_pg",
            name="calcular_indice_riesgo_climatico_node",
        ),
    ])


def create_inference_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=infer_climate_risk,
            inputs=["modelo_clasificacion", "climate_inference_input"],
            outputs="climate_inference_output",
            name="inferir_riesgo_climatico_node"
        ),
        node(
            func=calcular_indice_riesgo_climatico,
            inputs=["predicciones_climaticas", "run_id"],   #  ← AÑADE run_id
            outputs="risk_index_pg",
            name="calcular_indice_riesgo_climatico_node",
        ),
    ])
