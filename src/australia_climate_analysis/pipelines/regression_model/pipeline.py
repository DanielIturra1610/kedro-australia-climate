from kedro.pipeline import Pipeline, node
from . import nodes                              # ← alias “nodes”
from australia_climate_analysis.utils.run_id import new_run  # ← helper run_id


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        # 1️⃣ Identificador de ejecución
        node(
            func=new_run,
            inputs=None,
            outputs=["run_id", "metadata_runs"],
            name="register_run_node",
        ),

        # 2️⃣ Split
        node(
            func=nodes.split_data_for_regression,               # ← vía nodes.*
            inputs=["weather_with_rainfall_outliers",
                    "params:test_size",
                    "params:random_state"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_for_regression_node",
        ),

        # 3️⃣ Train
        node(
            func=nodes.train_regression_model,
            inputs=["X_train", "y_train"],
            outputs="regression_model",
            name="train_regression_model_node",
        ),

        # 4️⃣ Evaluate
        node(
            func=nodes.evaluate_regression_model,               # ← vía nodes.*
            inputs=["regression_model", "X_test", "y_test", "run_id"],
            outputs=["reg_metrics_local",          # (Memory / JSON local)
                     "regression_metrics_pg"],     # → tabla ml_metrics.regression
            name="evaluate_regression_model_node_simle",
        ),
    ])
