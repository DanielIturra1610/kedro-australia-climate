"""
Pipeline de Naive Bayes para predecir el clima de mañana basado en el clima de hoy.
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_data_for_naive_bayes,
    train_naive_bayes_model,
    evaluate_naive_bayes_model,
    predict_next_day_weather,
    predict_from_synthetic_data  # 🆕 NUEVA FUNCIÓN IMPORTADA
)

def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de Naive Bayes para predicción del clima.

    Returns:
        Pipeline: Pipeline de Naive Bayes para predicción del clima.
    """
    return pipeline(
        [
            node(
                func=prepare_data_for_naive_bayes,
                inputs=["weather_raw", "params:test_size", "params:random_state"],
                outputs=["X_train_naive_bayes", "X_test_naive_bayes", "y_train_naive_bayes", "y_test_naive_bayes"],
                name="prepare_data_for_naive_bayes",
            ),
            node(
                func=train_naive_bayes_model,
                inputs=["X_train_naive_bayes", "y_train_naive_bayes"],
                outputs="naive_bayes_model",
                name="train_naive_bayes_model",
            ),
            node(
                func=evaluate_naive_bayes_model,
                inputs=["naive_bayes_model", "X_test_naive_bayes", "y_test_naive_bayes", "params:run_id"],
                outputs=["naive_bayes_metrics_json", "naive_bayes_metrics_pg"],
                name="evaluate_naive_bayes_model",
            ),
            node(
                func=predict_next_day_weather,
                inputs=["naive_bayes_model", "weather_raw", "params:prediction_date"],
                outputs="next_day_prediction",
                name="predict_next_day_weather",
            ),
            # 🆕 NUEVO NODO PARA DATOS SINTÉTICOS
            node(
                func=predict_from_synthetic_data,
                inputs="synthetic_weather_data",
                outputs="naive_bayes_synthetic_prediction",
                name="predict_from_synthetic_data_node",
            ),
        ]
    )
