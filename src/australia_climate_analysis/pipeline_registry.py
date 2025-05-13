# pipeline_registry.py
from kedro.pipeline import Pipeline, pipeline
from australia_climate_analysis.pipelines.climate_risk import pipeline as climate_risk_pl
from australia_climate_analysis.pipelines.regression_model import pipeline as linear_pl
from australia_climate_analysis.pipelines.regression_tree import pipeline as tree_pl
from australia_climate_analysis.pipelines.regression_svm.pipeline import create_pipeline as svm_pipeline_pl  # Pipeline de SVM
from australia_climate_analysis.pipelines.regression_random_forest.pipeline import create_pipeline as random_forest_pipeline_pl  # Pipeline de Random Forest

def register_pipelines() -> dict[str, Pipeline]:
    # Los pipelines previos
    climate_risk = climate_risk_pl.create_pipeline()
    inference = climate_risk_pl.create_inference_pipeline()

    # Los pipelines de regresión
    regression_linear = linear_pl.create_pipeline()  # Pipeline de regresión lineal
    regression_tree = tree_pl.create_pipeline()  # Pipeline árbol de decisión
    regression_svm = svm_pipeline_pl()  # Pipeline de SVM
    regression_random_forest = random_forest_pipeline_pl()  # Pipeline de Random Forest

    # Se crea el pipeline completo
    full_pipeline = pipeline(
        climate_risk + regression_linear + regression_tree + regression_svm + regression_random_forest  # Se combinan todos los pipelines sin duplicados
    )

    return {
        "climate_risk": climate_risk,
        "inference": inference,
        "regression_model": regression_linear,  # Pipeline regresión lineal
        "regression_tree": regression_tree,  # Pipeline árbol de decisión
        "regression_svm": regression_svm,  # Pipeline SVM
        "regression_random_forest": regression_random_forest,  # Pipeline Random Forest
        "__default__": full_pipeline,  # Pipeline completo
    }

