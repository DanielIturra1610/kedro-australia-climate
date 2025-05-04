from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from australia_climate_analysis.pipelines.regression_model import pipeline as regression_model_pipeline
from australia_climate_analysis.pipelines.regression_model import pipeline as regression_pipeline   
from australia_climate_analysis.pipelines.climate_risk import (
    pipeline as climate_risk_pipeline,
)
from kedro.pipeline import pipeline

def register_pipelines() -> dict[str, Pipeline]:
    climate_risk = climate_risk_pipeline.create_pipeline()
    inference = climate_risk_pipeline.create_inference_pipeline()
    regression_model = regression_model_pipeline.create_pipeline()
    regression = regression_pipeline.create_pipeline()

    # Pipeline combinado: primero riesgo climático, luego regresión
    full_pipeline = pipeline(
        climate_risk + regression
    )

    return {
        "climate_risk": climate_risk,
        "inference": inference,
        "regression_model": regression_model,
        "regression": regression,
        "__default__": full_pipeline,
    }   