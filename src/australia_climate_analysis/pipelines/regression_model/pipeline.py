from kedro.pipeline import Pipeline, node
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=nodes.split_data_for_regression,
            inputs=["weather_with_rainfall_outliers", "params:test_size", "params:random_state"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_for_regression_node",
        ),
        node(
            func=nodes.train_regression_model,
            inputs=["X_train", "y_train"],
            outputs="regression_model",
            name="train_regression_model_node",
        ),
        node(
            func=nodes.evaluate_regression_model,
            inputs=["regression_model", "X_test", "y_test"],
            outputs={
                "regression_model_metrics": "regression_model_metrics",  # Saves dict to JSON
                "regression_model_metrics_pg": "regression_model_metrics_pg"  # Saves DataFrame to SQL
            },
            name="evaluate_regression_model_node"
        )
    ])
