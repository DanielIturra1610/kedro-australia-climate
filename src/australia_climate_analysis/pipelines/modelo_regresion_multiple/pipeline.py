from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_multiple_regression_model, evaluate_multiple_regression_model
from australia_climate_analysis.pipelines.regression_model.nodes import split_data_for_regression

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data_for_regression,
                inputs=["weather_data_with_features", "params:test_size", "params:random_state"],
                outputs=["X_train_multiple", "X_test_multiple", "y_train_multiple", "y_test_multiple"],
                name="split_data_for_multiple_regression",
            ),
            node(
                func=train_multiple_regression_model,
                inputs=["X_train_multiple", "y_train_multiple"],
                outputs="multiple_regression_model",
                name="train_multiple_regression_model",
            ),
            node(
                func=evaluate_multiple_regression_model,
                inputs=["multiple_regression_model", "X_test_multiple", "y_test_multiple", "params:run_id"],
                outputs=["multiple_regression_metrics_json", "multiple_regression_metrics_pg"],
                name="evaluate_multiple_regression_model",
            ),
        ]
    )