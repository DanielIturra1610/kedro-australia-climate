from kedro.pipeline import Pipeline, node, pipeline
from .result_interpretation import (
    generate_model_comparison_report,
    generate_solution_proposals
)
from .metrics_collector import (
    combine_regression_metrics,
    combine_classification_metrics,
    generate_model_performance_summary
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline para interpretación de resultados y propuestas de solución.
    """
    return pipeline([
        # Combinar métricas de modelos de regresión
        node(
            func=combine_regression_metrics,
            inputs={
                "svm_metrics": "svm_metrics",
                "random_forest_metrics": "random_forest_metrics",
                "multiple_regression_metrics": "multiple_regression_metrics",
                "regression_linear_metrics": "regression_linear_metrics",
                "regression_tree_metrics": "regression_tree_metrics"
            },
            outputs="regression_metrics_combined",
            name="combine_regression_metrics_node",
        ),
        
        # Combinar métricas de modelos de clasificación
        node(
            func=combine_classification_metrics,
            inputs={
                "naive_bayes_metrics": "naive_bayes_metrics",
                "climate_risk_metrics": "climate_risk_metrics"
            },
            outputs="classification_metrics_combined",
            name="combine_classification_metrics_node",
        ),
        
        # Generar resumen comparativo de rendimiento de modelos
        node(
            func=generate_model_performance_summary,
            inputs={
                "regression_metrics": "regression_metrics_combined",
                "classification_metrics": "classification_metrics_combined"
            },
            outputs="models_performance_summary",
            name="generate_model_performance_summary_node",
        ),
        
        # Generación de reporte interpretativo comparando todos los modelos
        node(
            func=generate_model_comparison_report,
            inputs={
                "regression_metrics": "regression_metrics_combined",
                "classification_metrics": "classification_metrics_combined",
                "unsupervised_insights": "unsupervised_insights",
                "output_dir": "params:interpretation_output_dir"
            },
            outputs="model_comparison_report",
            name="generate_model_comparison_report_node",
        ),
        
        # Generación de propuestas de solución basadas en los insights
        node(
            func=generate_solution_proposals,
            inputs={
                "interpretation_report": "model_comparison_report",
                "climate_data": "df_final",
                "unsupervised_insights": "unsupervised_insights"
            },
            outputs="solution_proposals",
            name="generate_solution_proposals_node",
        )
    ])