# pipeline_registry.py
from kedro.pipeline import Pipeline, pipeline
from australia_climate_analysis.pipelines.climate_risk import pipeline as climate_risk_pl
from australia_climate_analysis.pipelines.regression_model import pipeline as linear_pl
from australia_climate_analysis.pipelines.regression_tree import pipeline as tree_pl
from australia_climate_analysis.pipelines.regression_svm.pipeline import create_pipeline as svm_pipeline_pl  # Pipeline de SVM
from australia_climate_analysis.pipelines.regression_random_forest.pipeline import create_pipeline as random_forest_pipeline_pl  # Pipeline de Random Forest
from australia_climate_analysis.pipelines.modelo_regresion_multiple.pipeline import create_pipeline as multiple_regression_pipeline_pl  # Pipeline de Regresión Múltiple
from australia_climate_analysis.pipelines.data_engineering.pipeline import create_pipeline as data_engineering_pipeline_pl  # Pipeline de Ingeniería de Datos
from australia_climate_analysis.pipelines.naive_bayes_forecast.pipeline import create_pipeline as naive_bayes_pipeline_pl  # Pipeline de Naive Bayes para predicción del clima

def register_pipelines() -> dict[str, Pipeline]:
    # Los pipelines previos
    climate_risk = climate_risk_pl.create_pipeline()
    inference = climate_risk_pl.create_inference_pipeline()
    
    # Pipeline de ingeniería de datos
    data_engineering = data_engineering_pipeline_pl()

    # Los pipelines de regresión
    regression_linear = linear_pl.create_pipeline()  # Pipeline de regresión lineal
    regression_tree = tree_pl.create_pipeline()  # Pipeline árbol de decisión
    regression_svm = svm_pipeline_pl()  # Pipeline de SVM
    regression_random_forest = random_forest_pipeline_pl()  # Pipeline de Random Forest
    regression_multiple = multiple_regression_pipeline_pl()  # Pipeline de Regresión Múltiple
    
    # Pipeline de Naive Bayes para predicción del clima
    naive_bayes_forecast = naive_bayes_pipeline_pl()  # Pipeline de Naive Bayes

    # Se crea el pipeline completo
    full_pipeline = pipeline(
        data_engineering + climate_risk + regression_linear + regression_tree + regression_svm + 
        regression_random_forest + regression_multiple + naive_bayes_forecast  # Se combinan todos los pipelines sin duplicados
    )

    return {
        "climate_risk": climate_risk,
        "inference": inference,
        "data_engineering": data_engineering,  # Pipeline ingeniería de datos
        "regression_model": regression_linear,  # Pipeline regresión lineal
        "regression_tree": regression_tree,  # Pipeline árbol de decisión
        "regression_svm": regression_svm,  # Pipeline SVM
        "regression_random_forest": regression_random_forest,  # Pipeline Random Forest
        "regression_multiple": regression_multiple,  # Pipeline Regresión Múltiple
        "modelo_regresion_multiple": regression_multiple,  # Alias para el mismo pipeline
        "naive_bayes_forecast": naive_bayes_forecast,  # Pipeline de Naive Bayes para predicción del clima
        "__default__": full_pipeline,  # Pipeline completo
    }

