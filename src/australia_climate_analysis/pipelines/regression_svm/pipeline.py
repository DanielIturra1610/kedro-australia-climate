from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_svm_regression_model, evaluate_svm_regression_model
from australia_climate_analysis.pipelines.regression_model.nodes import split_data_for_regression

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data_for_regression,
                inputs=["weather_data_with_features", "params:test_size", "params:random_state"],
                outputs=["X_train_svm", "X_test_svm", "y_train_svm", "y_test_svm"],
                name="split_data_for_svm_regression",
            ),
            node(
                func=train_svm_regression_model,
                inputs=["X_train_svm", "y_train_svm"],
                outputs="svm_model",
                name="train_svm_regression_model",
            ),
            node(
                func=evaluate_svm_regression_model,
                inputs=["svm_model", "X_test_svm", "y_test_svm", "params:run_id"],
                outputs=["svm_metrics_json", "svm_metrics_pg"],
                name="evaluate_svm_regression_model",
            ),
        ]
    )