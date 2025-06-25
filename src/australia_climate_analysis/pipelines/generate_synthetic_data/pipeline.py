from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_multiple_synthetic_days, generate_forecast_sequences

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_multiple_synthetic_days,
                inputs=["weather_raw", "params:num_days", "params:start_date"],
                outputs="synthetic_weather_data",
                name="generate_synthetic_day_node",
            ),
            # Nuevo nodo para generar secuencias de 3 días para predicción
            node(
                func=generate_forecast_sequences,
                inputs=["weather_raw", "params:forecast_locations", "params:forecast_start_date", "params:forecast_sequence_length"],
                outputs="forecast_sequences_data",
                name="generate_forecast_sequences_node",
            )
        ]
    )