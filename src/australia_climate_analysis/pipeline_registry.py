# pipeline_registry.py
from kedro.pipeline import Pipeline, pipeline
from australia_climate_analysis.pipelines.climate_risk import pipeline as climate_risk_pl
from australia_climate_analysis.pipelines.regression_model import pipeline as linear_pl
from australia_climate_analysis.pipelines.regression_tree import pipeline as tree_pl
from australia_climate_analysis.pipelines.regression_svm.pipeline import create_pipeline as svm_pipeline_pl
from australia_climate_analysis.pipelines.regression_random_forest.pipeline import create_pipeline as random_forest_pipeline_pl
from australia_climate_analysis.pipelines.modelo_regresion_multiple.pipeline import create_pipeline as multiple_regression_pipeline_pl
from australia_climate_analysis.pipelines.data_engineering.pipeline import create_pipeline as data_engineering_pipeline_pl
from australia_climate_analysis.pipelines.naive_bayes_forecast.pipeline import create_pipeline as naive_bayes_pipeline_pl
from australia_climate_analysis.pipelines.generate_synthetic_data.pipeline import create_pipeline as synthetic_data_pipeline_pl  # 🆕
from australia_climate_analysis.pipelines.regression_svm.pipeline import create_pipeline as svm_pipeline_pl  # Pipeline de SVM
from australia_climate_analysis.pipelines.regression_random_forest.pipeline import create_pipeline as random_forest_pipeline_pl  # Pipeline de Random Forest
from australia_climate_analysis.pipelines.modelo_regresion_multiple.pipeline import create_pipeline as multiple_regression_pipeline_pl  # Pipeline de Regresión Múltiple
from australia_climate_analysis.pipelines.data_engineering.pipeline import create_pipeline as data_engineering_pipeline_pl  # Pipeline de Ingeniería de Datos
from australia_climate_analysis.pipelines.naive_bayes_forecast.pipeline import create_pipeline as naive_bayes_pipeline_pl  # Pipeline de Naive Bayes para predicción del clima
from australia_climate_analysis.pipelines.unsupervised_models.pipeline import create_pipeline as unsupervised_pipeline_pl  # Pipeline de modelos no supervisados
from australia_climate_analysis.pipelines.interpretation.pipeline import create_pipeline as interpretation_pipeline_pl  # Pipeline de interpretación y propuestas

def register_pipelines() -> dict[str, Pipeline]:
    # Pipelines existentes
    climate_risk = climate_risk_pl.create_pipeline()
    inference = climate_risk_pl.create_inference_pipeline()
    data_engineering = data_engineering_pipeline_pl()
    regression_linear = linear_pl.create_pipeline()
    regression_tree = tree_pl.create_pipeline()
    regression_svm = svm_pipeline_pl()
    regression_random_forest = random_forest_pipeline_pl()
    regression_multiple = multiple_regression_pipeline_pl()
    naive_bayes_forecast = naive_bayes_pipeline_pl()
    
    # 🆕 Pipeline de generación de datos sintéticos
    synthetic_data = synthetic_data_pipeline_pl()
    # Pipeline de Naive Bayes para predicción del clima
    naive_bayes_forecast = naive_bayes_pipeline_pl()  # Pipeline de Naive Bayes
    
    # Nuevos pipelines: modelos no supervisados e interpretación
    unsupervised_models = unsupervised_pipeline_pl()  # Pipeline de modelos no supervisados
    interpretation = interpretation_pipeline_pl()  # Pipeline de interpretación y propuestas

    # Pipeline completo combinado
    full_pipeline = pipeline(
        data_engineering + climate_risk + regression_linear + regression_tree + regression_svm +
        regression_random_forest + regression_multiple + naive_bayes_forecast + synthetic_data +
        unsupervised_models + interpretation  # Se agregan los nuevos pipelines
    )
    
    # Se crea un pipeline avanzado específico para análisis no supervisados e interpretación
    advanced_pipeline = pipeline(
        data_engineering + unsupervised_models + interpretation
    )

    return {
        "climate_risk": climate_risk,
        "inference": inference,
        "data_engineering": data_engineering,
        "regression_model": regression_linear,
        "regression_tree": regression_tree,
        "regression_svm": regression_svm,
        "regression_random_forest": regression_random_forest,
        "regression_multiple": regression_multiple,
        "modelo_regresion_multiple": regression_multiple,
        "naive_bayes_forecast": naive_bayes_forecast,
        "generate_synthetic_data": synthetic_data,
        "unsupervised_models": unsupervised_models,
        "interpretation": interpretation,
        "advanced_analysis": advanced_pipeline,
    }
