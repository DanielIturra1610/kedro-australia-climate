from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_random_forest_model, evaluate_random_forest_model
from australia_climate_analysis.pipelines.regression_model.nodes import split_data_for_regression

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data_for_regression,
                inputs=["weather_data_with_features", "params:test_size", "params:random_state"],
                outputs=["X_train_rf", "X_test_rf", "y_train_rf", "y_test_rf"],
                name="split_data_for_random_forest",
            ),
            node(
                func=train_random_forest_model,
                inputs=["X_train_rf", "y_train_rf"],
                outputs="random_forest_model",
                name="train_random_forest_model",
            ),
            node(
                func=evaluate_random_forest_model,
                inputs=["random_forest_model", "X_test_rf", "y_test_rf", "params:run_id"],
                outputs=["random_forest_metrics_json", "random_forest_metrics_pg"],
                name="evaluate_random_forest_model",
            ),
        ]
    )