from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_forecast_features,
    train_forecast_models,
    generate_forecast,
    evaluate_forecast_accuracy,
    prepare_forecast_for_api,
    save_forecast_to_database
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Nodo para preparar los datos de entrenamiento
            node(
                func=prepare_forecast_features,
                inputs="weather_raw",
                outputs=["forecast_features", "forecast_targets"],
                name="prepare_forecast_features_node",
            ),
            # Nodo para entrenar los modelos de predicción
            node(
                func=train_forecast_models,
                inputs=["forecast_features", "forecast_targets", "params:forecast_model"],
                outputs="forecast_model_trained",
                name="train_forecast_models_node",
            ),
            # Nodo para generar pronósticos a partir de datos iniciales
            node(
                func=generate_forecast,
                inputs=["forecast_model_trained", "synthetic_weather_data", "params:forecast_sequence_length"],
                outputs="weather_forecast_3days",
                name="generate_forecast_node",
            ),
            # Nodo para evaluar la precisión del pronóstico (opcional si tenemos datos reales)
            node(
                func=evaluate_forecast_accuracy,
                inputs=["weather_forecast_3days", "synthetic_weather_data"],
                outputs="forecast_metrics",
                name="evaluate_forecast_accuracy_node",
            ),
            # Nodo para preparar datos para la API
            node(
                func=prepare_forecast_for_api,
                inputs="weather_forecast_3days",
                outputs="weather_forecast_api_format",
                name="prepare_forecast_for_api_node",
            ),
            node(
                func=save_forecast_to_database,
                inputs=["weather_forecast_api_format", "params:database"],
                outputs="forecast_db_result",
                name="save_forecast_to_database_node",
            ),
        ]
    )