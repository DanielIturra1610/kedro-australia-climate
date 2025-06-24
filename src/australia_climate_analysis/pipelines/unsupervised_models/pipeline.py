from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocess_data_for_clustering,
    apply_kmeans_clustering,
    apply_dbscan_clustering,
    apply_pca_dimensionality_reduction,
    apply_isolation_forest,
    generate_unsupervised_insights
)

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de modelos no supervisados para análisis climático.
    """
    return pipeline([
        # Preprocesamiento para modelos no supervisados
        node(
            func=preprocess_data_for_clustering,
            inputs="df_final",  # Usar los datos del pipeline de data_engineering
            outputs="data_for_unsupervised",
            name="preprocess_data_for_unsupervised_node",
        ),
        
        # Aplicar K-means clustering
        node(
            func=apply_kmeans_clustering,
            inputs=["data_for_unsupervised", "params:kmeans_params"],
            outputs=["kmeans_model", "kmeans_clusters", "kmeans_metrics"],
            name="apply_kmeans_node"
        ),
        
        # Aplicar DBSCAN para detección de anomalías
        node(
            func=apply_dbscan_clustering,
            inputs=["data_for_unsupervised", "params:dbscan_params"],
            outputs=["dbscan_clusters", "dbscan_metrics"],
            name="apply_dbscan_node",
        ),
        
        # Aplicar PCA para reducción de dimensionalidad
        node(
            func=apply_pca_dimensionality_reduction,
            inputs=["data_for_unsupervised", "params:pca_params"],
            outputs=["pca_model", "pca_reduced_data", "pca_metrics"],
            name="apply_pca_node",
        ),
        
        # Aplicar Isolation Forest para detección de anomalías
        node(
            func=apply_isolation_forest,
            inputs=["data_for_unsupervised", "params:isolation_forest_params"],
            outputs=["isolation_df", "isolation_metrics"],
            name="apply_isolation_forest_node",
        ),
        
        # Generar insights basados en todos los modelos
        node(
            func=generate_unsupervised_insights,
            inputs=[
                "kmeans_clusters", 
                "dbscan_clusters", 
                "pca_reduced_data", 
                "isolation_df",
                "kmeans_metrics",
                "dbscan_metrics",
                "pca_metrics",
                "isolation_metrics"
            ],
            outputs="unsupervised_insights",
            name="generate_unsupervised_insights_node",
        )
    ])