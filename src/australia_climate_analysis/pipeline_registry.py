"""Pipeline registry"""
from kedro.pipeline import Pipeline, pipeline

# Importaciones de los pipelines existentes
from australia_climate_analysis.pipelines.climate_risk import pipeline as climate_risk_pl
from australia_climate_analysis.pipelines.regression_model import pipeline as linear_pl
from australia_climate_analysis.pipelines.regression_tree import pipeline as tree_pl
from australia_climate_analysis.pipelines.regression_svm.pipeline import create_pipeline as svm_pipeline_pl
from australia_climate_analysis.pipelines.regression_random_forest.pipeline import create_pipeline as random_forest_pipeline_pl
from australia_climate_analysis.pipelines.modelo_regresion_multiple.pipeline import create_pipeline as multiple_regression_pipeline_pl
from australia_climate_analysis.pipelines.data_engineering.pipeline import create_pipeline as data_engineering_pipeline_pl
from australia_climate_analysis.pipelines.naive_bayes_forecast.pipeline import create_pipeline as naive_bayes_pipeline_pl
from australia_climate_analysis.pipelines.generate_synthetic_data.pipeline import create_pipeline as synthetic_data_pipeline_pl
from australia_climate_analysis.pipelines.unsupervised_models.pipeline import create_pipeline as unsupervised_pipeline_pl
from australia_climate_analysis.pipelines.interpretation.pipeline import create_pipeline as interpretation_pipeline_pl
from australia_climate_analysis.pipelines.weather_forecast_3days.pipeline import create_pipeline as weather_forecast_3days_pipeline_pl


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
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
    
    # Pipeline de generación de datos sintéticos
    synthetic_data = synthetic_data_pipeline_pl()
    
    # Pipelines de modelos no supervisados e interpretación
    unsupervised_models = unsupervised_pipeline_pl()
    interpretation = interpretation_pipeline_pl()

    # Pipeline de pronóstico de 3 días
    weather_forecast_3days = weather_forecast_3days_pipeline_pl()
    
    # Pipeline completo combinado
    full_pipeline = pipeline(
        data_engineering + climate_risk + regression_linear + regression_tree + regression_svm +
        regression_random_forest + regression_multiple + naive_bayes_forecast + synthetic_data +
        unsupervised_models + interpretation + weather_forecast_3days
    )
    
    # Pipeline avanzado específico para análisis no supervisados e interpretación
    advanced_pipeline = pipeline(
        data_engineering + unsupervised_models + interpretation
    )

    return {
        "__default__": full_pipeline,
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
        "weather_forecast_3days": weather_forecast_3days,
    }