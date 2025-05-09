from kedro.pipeline import Pipeline, node
from . import nodes
from australia_climate_analysis.utils.run_id import new_run


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=new_run,
            inputs=None,
            outputs=["run_id", "metadata_runs"],
            name="register_run_node",
        ),
        node(
            func=nodes.split_data_for_regression,
            inputs=["weather_with_rainfall_outliers",
                    "params:test_size",
                    "params:random_state"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_for_regression_node",
        ),
        node(
            func=nodes.train_regression_model,
            inputs=["X_train", "y_train"],
            outputs="tree_model",
            name="train_regression_tree_node",
        ),
        node(
            func=nodes.evaluate_regression_model,
            inputs=["tree_model", "X_test", "y_test", "run_id"],
            outputs=["tree_metrics_local",          # (local/Memory)
                     "regression_metrics_pg"],      # ✔ misma tabla
            name="evaluate_regression_tree_node",
        ),
    ])
