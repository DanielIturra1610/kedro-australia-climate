# pipelines/regression_random_forest/pipeline.py
from kedro.pipeline import Pipeline, node
from .random_forest import train_random_forest_model, evaluate_random_forest_model

def create_pipeline():
    """Define the Random Forest pipeline."""
    return Pipeline(
        [
            node(
                func=train_random_forest_model,
                inputs=["X_train", "y_train"],
                outputs="random_forest_model",
                name="train_random_forest_model_node",
            ),
            node(
                func=evaluate_random_forest_model,
                inputs=["random_forest_model", "X_test", "y_test"],
                outputs=["random_forest_metrics_json", "random_forest_metrics_long"],
                name="evaluate_random_forest_model_node",
            ),
        ]
    )
