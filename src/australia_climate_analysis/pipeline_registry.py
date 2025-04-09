from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from australia_climate_analysis.pipelines.climate_risk import (
    pipeline as climate_risk_pipeline,
)

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "climate_risk": climate_risk_pipeline.create_pipeline(),
        "inference": climate_risk_pipeline.create_inference_pipeline(),
        "__default__": climate_risk_pipeline.create_pipeline(),
    }