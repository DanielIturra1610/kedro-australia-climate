# pipelines/regression_svm/pipeline.py
from kedro.pipeline import Pipeline, node
from .svm import train_svm_model, evaluate_svm_model

def create_pipeline():
    """Define the SVM pipeline."""
    return Pipeline(
        [
            node(
                func=train_svm_model,
                inputs=["X_train", "y_train"],
                outputs="svm_model",
                name="train_svm_model_node",
            ),
            node(
                func=evaluate_svm_model,
                inputs=["svm_model", "X_test", "y_test"],
                outputs=["svm_metrics_json", "svm_metrics_long"],
                name="evaluate_svm_model_node",
            ),
        ]
    )
