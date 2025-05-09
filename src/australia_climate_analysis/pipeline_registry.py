from kedro.pipeline import Pipeline, pipeline
from australia_climate_analysis.pipelines.climate_risk import pipeline as climate_risk_pl
from australia_climate_analysis.pipelines.regression_model import pipeline as linear_pl
from australia_climate_analysis.pipelines.regression_tree  import pipeline as tree_pl

def register_pipelines() -> dict[str, Pipeline]:
    climate_risk      = climate_risk_pl.create_pipeline()
    inference         = climate_risk_pl.create_inference_pipeline()

    regression_linear = linear_pl.create_pipeline()
    regression_tree   = tree_pl.create_pipeline()   # ← se registra, pero no se suma

    full_pipeline = pipeline(
        climate_risk + regression_linear            # ← solo uno, sin duplicados
    )

    return {
        "climate_risk"    : climate_risk,
        "inference"       : inference,
        "regression_model": regression_linear,   # Lineal
        "regression_tree" : regression_tree,     # Árbol
        "__default__"     : full_pipeline,
    }
