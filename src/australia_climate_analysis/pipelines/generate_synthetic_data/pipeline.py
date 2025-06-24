from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_multiple_synthetic_days

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=generate_multiple_synthetic_days,
                inputs=["weather_raw", "params:num_days", "params:start_date"],
                outputs="synthetic_weather_data",
                name="generate_synthetic_day_node",
            )
        ]
    )



