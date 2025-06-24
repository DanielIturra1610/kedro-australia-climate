"""
Nodos de modelos no supervisados para análisis climático.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import logging
import time
import os
from typing import Dict, Tuple, Any, List

logger = logging.getLogger(__name__)

def preprocess_data_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesa los datos para algoritmos de clustering.
    
    Args:
        df: DataFrame con datos meteorológicos
        
    Returns:
        DataFrame con datos estandarizados y preparados para clustering
    """
    logger.info("Preprocesando datos para clustering")
    
    # Seleccionar características numéricas relevantes
    numeric_features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Temp9am', 'Temp3pm'
    ]
    
    # Filtrar solo las columnas que existen en el DataFrame
    available_features = [col for col in numeric_features if col in df.columns]
    
    # Crear una copia de trabajo
    X = df[available_features].copy()
    
    # Imputar valores faltantes con la mediana
    for col in available_features:
        X[col] = X[col].fillna(X[col].median())
    
    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Crear DataFrame con datos escalados
    X_scaled_df = pd.DataFrame(X_scaled, columns=available_features)
    
    # Agregar ubicación para análisis posterior si existe
    if 'Location' in df.columns:
        X_scaled_df['Location'] = df['Location'].values
    
    logger.info(f"Datos preprocesados: {X_scaled_df.shape[0]} registros, {X_scaled_df.shape[1]} características")
    return X_scaled_df

def apply_kmeans_clustering(df: pd.DataFrame, params: Dict) -> Tuple[KMeans, pd.DataFrame, Dict]:
    """
    Aplica clustering K-means a los datos climáticos.
    
    Args:
        df: DataFrame con datos preprocesados
        params: Diccionario con parámetros para K-means {'n_clusters': int}
        
    Returns:
        modelo KMeans entrenado, DataFrame con asignaciones de cluster y métricas del modelo
    """
    start_time = time.time()
    n_clusters = params.get('n_clusters', 4)
    logger.info(f"Aplicando K-means clustering con {n_clusters} clusters")
    
    # Extraer características numéricas (excluyendo 'Location' si existe)
    if 'Location' in df.columns:
        location = df['Location'].copy()
        X = df.drop(columns=['Location'])
    else:
        location = None
        X = df.copy()
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calcular métricas
    silhouette = silhouette_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
    inertia = kmeans.inertia_
    
    # Crear DataFrame con resultados
    result_df = df.copy()
    result_df['cluster'] = cluster_labels
    
    # Obtener centroides para interpretación
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
    
    # Tiempo de ejecución
    execution_time = time.time() - start_time
    
    # Crear diccionario con métricas
    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": silhouette,
        "inertia": inertia,
        "execution_time": execution_time,
        "centroids": centroids.to_dict(orient='records')
    }
    
    logger.info(f"K-means completado en {execution_time:.2f} segundos")
    logger.info(f"Silhouette score: {silhouette:.4f}")
    
    # Distribución de clusters
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        logger.info(f"Cluster {cluster_id}: {count} registros ({count/len(cluster_labels)*100:.2f}%)")
    
    return kmeans, result_df, metrics

def apply_dbscan_clustering(df: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Aplica clustering DBSCAN para detección de grupos de densidad y outliers.
    
    Args:
        df: DataFrame con datos preprocesados
        params: Diccionario con parámetros para DBSCAN {'eps': float, 'min_samples': int}
        
    Returns:
        DataFrame con asignaciones de cluster y métricas del modelo
    """
    start_time = time.time()
    eps = params.get('eps', 0.5)
    min_samples = params.get('min_samples', 5)
    logger.info(f"Aplicando DBSCAN clustering con eps={eps}, min_samples={min_samples}")
    
    # Extraer características numéricas (excluyendo 'Location' si existe)
    if 'Location' in df.columns:
        location = df['Location'].copy()
        X = df.drop(columns=['Location'])
    else:
        location = None
        X = df.copy()
    
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)
    
    # Calcular métricas
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    silhouette = silhouette_score(X, cluster_labels) if len(np.unique(cluster_labels)) > 1 and -1 not in cluster_labels else 0
    
    # Crear DataFrame con resultados
    result_df = df.copy()
    result_df['dbscan_cluster'] = cluster_labels
    result_df['is_anomaly'] = (cluster_labels == -1).astype(int)
    
    # Tiempo de ejecución
    execution_time = time.time() - start_time
    
    # Crear diccionario con métricas
    metrics = {
        "n_clusters": n_clusters,
        "n_noise_points": n_noise,
        "noise_percentage": n_noise / len(cluster_labels) * 100,
        "silhouette_score": silhouette,
        "execution_time": execution_time
    }
    
    logger.info(f"DBSCAN completado en {execution_time:.2f} segundos")
    logger.info(f"Número de clusters: {n_clusters}")
    logger.info(f"Puntos de ruido (anomalías): {n_noise} ({metrics['noise_percentage']:.2f}%)")
    
    # Distribución de clusters
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        cluster_name = "Anomalía" if cluster_id == -1 else f"Cluster {cluster_id}"
        logger.info(f"{cluster_name}: {count} registros ({count/len(cluster_labels)*100:.2f}%)")
    
    return result_df, metrics

def apply_pca_dimensionality_reduction(df: pd.DataFrame, params: Dict) -> Tuple[PCA, pd.DataFrame, Dict]:
    """
    Aplica PCA para reducción de dimensionalidad y análisis de componentes.
    
    Args:
        df: DataFrame con datos preprocesados
        params: Diccionario con parámetros para PCA {'n_components': int}
        
    Returns:
        modelo PCA entrenado, DataFrame con componentes principales y métricas del modelo
    """
    start_time = time.time()
    n_components = params.get('n_components', 2)
    logger.info(f"Aplicando PCA con {n_components} componentes")
    
    # Extraer características numéricas (excluyendo 'Location' si existe)
    if 'Location' in df.columns:
        location = df['Location'].copy()
        X = df.drop(columns=['Location'])
    else:
        location = None
        X = df.copy()
    
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X)
    
    # Crear DataFrame con resultados
    result_columns = [f'PC{i+1}' for i in range(n_components)]
    result_df = pd.DataFrame(data=principal_components, columns=result_columns)
    
    # Agregar ubicación si existe
    if location is not None:
        result_df['Location'] = location.values
    
    # Calcular varianza explicada
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Crear matriz de contribuciones de características
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=result_columns,
        index=X.columns
    )
    
    # Tiempo de ejecución
    execution_time = time.time() - start_time
    
    # Crear diccionario con métricas
    metrics = {
        "n_components": n_components,
        "explained_variance_ratio": explained_variance.tolist(),
        "cumulative_explained_variance": cumulative_variance.tolist(),
        "feature_importance": feature_importance.to_dict(),
        "execution_time": execution_time
    }
    
    logger.info(f"PCA completado en {execution_time:.2f} segundos")
    logger.info(f"Varianza explicada por componentes: {[f'{var:.4f}' for var in explained_variance]}")
    logger.info(f"Varianza acumulada: {cumulative_variance[-1]:.4f}")
    
    return pca, result_df, metrics

def apply_isolation_forest(df: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Aplica Isolation Forest para detección de anomalías climáticas.
    
    Args:
        df: DataFrame con datos preprocesados
        params: Diccionario con parámetros para Isolation Forest {'contamination': float}
        
    Returns:
        DataFrame con indicadores de anomalías y métricas del modelo
    """
    start_time = time.time()
    contamination = params.get('contamination', 0.05)
    logger.info(f"Aplicando Isolation Forest con contaminación {contamination}")
    
    # Extraer características numéricas (excluyendo 'Location' si existe)
    if 'Location' in df.columns:
        location = df['Location'].copy()
        X = df.drop(columns=['Location'])
    else:
        location = None
        X = df.copy()
    
    # Aplicar Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X)
    anomaly_scores = iso_forest.decision_function(X)
    
    # Convertir predicciones (-1 para anomalías, 1 para normales) a formato más intuitivo
    # 1 para anomalías, 0 para normales
    is_anomaly = (anomaly_labels == -1).astype(int)
    
    # Crear DataFrame con resultados
    result_df = df.copy()
    result_df['anomaly_score'] = anomaly_scores
    result_df['is_anomaly'] = is_anomaly
    
    # Calcular métricas
    n_anomalies = sum(is_anomaly)
    
    # Tiempo de ejecución
    execution_time = time.time() - start_time
    
    # Crear diccionario con métricas
    metrics = {
        "n_anomalies": int(n_anomalies),
        "anomaly_percentage": float(n_anomalies / len(is_anomaly) * 100),
        "contamination": contamination,
        "execution_time": execution_time
    }
    
    logger.info(f"Isolation Forest completado en {execution_time:.2f} segundos")
    logger.info(f"Anomalías detectadas: {n_anomalies} ({metrics['anomaly_percentage']:.2f}%)")
    
    return result_df, metrics

def generate_unsupervised_insights(
    kmeans_df: pd.DataFrame, 
    dbscan_df: pd.DataFrame, 
    pca_df: pd.DataFrame,
    isolation_df: pd.DataFrame,
    kmeans_metrics: Dict,
    dbscan_metrics: Dict,
    pca_metrics: Dict,
    isolation_metrics: Dict
) -> Dict:
    """
    Genera insights basados en los resultados de los modelos no supervisados.
    
    Args:
        Resultados y métricas de los diferentes modelos
        
    Returns:
        Diccionario con insights y patrones identificados
    """
    logger.info("Generando insights de modelos no supervisados")
    
    insights = {
        "kmeans_insights": {},
        "dbscan_insights": {},
        "pca_insights": {},
        "anomaly_insights": {},
        "combined_insights": []
    }
    
    # 1. Insights de K-means - Características de cada cluster
    if 'cluster' in kmeans_df.columns:
        # Analizar características de cada cluster
        cluster_profiles = kmeans_df.groupby('cluster').mean()
        
        for cluster_id in sorted(kmeans_df['cluster'].unique()):
            # Determinar características distintivas del cluster
            cluster_data = cluster_profiles.loc[cluster_id]
            
                        # Ordenar características por su desviación respecto a la media global
            global_mean = kmeans_df.drop(columns=['cluster', 'Location'] if 'Location' in kmeans_df.columns else ['cluster']).mean()
            relative_importance = (cluster_data - global_mean).abs().sort_values(ascending=False)
            
            # Seleccionar las 3 características más distintivas
            top_features = relative_importance.head(3)
            
            # Determinar si los valores son altos o bajos comparados con la media
            feature_descriptions = []
            for feature, value in top_features.items():
                direction = "alto" if cluster_data[feature] > global_mean[feature] else "bajo"
                deviation = abs(cluster_data[feature] - global_mean[feature]) / global_mean[feature] * 100
                feature_descriptions.append(f"{feature} {direction} ({deviation:.1f}% desviación)")
            
            # Calcular tamaño del cluster
            cluster_size = (kmeans_df['cluster'] == cluster_id).sum()
            cluster_percentage = cluster_size / len(kmeans_df) * 100
            
            # Guardar insight para este cluster
            insights["kmeans_insights"][f"cluster_{cluster_id}"] = {
                "size": int(cluster_size),
                "percentage": float(cluster_percentage),
                "distinctive_features": feature_descriptions,
                "profile": cluster_data.to_dict()
            }
            
            # Crear descripción textual
            description = f"Cluster {cluster_id} ({cluster_percentage:.1f}% de registros): Caracterizado por {', '.join(feature_descriptions)}"
            insights["combined_insights"].append(description)
    
    # 2. Insights de DBSCAN - Análisis de anomalías y patrones densos
    if 'dbscan_cluster' in dbscan_df.columns:
        # Analizar clusters normales vs anomalías (-1)
        normal_clusters = dbscan_df[dbscan_df['dbscan_cluster'] != -1]
        anomalies = dbscan_df[dbscan_df['dbscan_cluster'] == -1]
        
        # Porcentaje de anomalías
        anomaly_percentage = len(anomalies) / len(dbscan_df) * 100
        
        # Analizar características de anomalías vs registros normales
        if len(anomalies) > 0 and len(normal_clusters) > 0:
            anomaly_means = anomalies.drop(columns=['dbscan_cluster', 'is_anomaly', 'Location'] 
                                        if 'Location' in anomalies.columns 
                                        else ['dbscan_cluster', 'is_anomaly']).mean()
            
            normal_means = normal_clusters.drop(columns=['dbscan_cluster', 'is_anomaly', 'Location']
                                            if 'Location' in normal_clusters.columns
                                            else ['dbscan_cluster', 'is_anomaly']).mean()
            
            # Encontrar las características con mayor diferencia
            differences = (anomaly_means - normal_means).abs().sort_values(ascending=False)
            top_diff_features = differences.head(3)
            
            diff_descriptions = []
            for feature, value in top_diff_features.items():
                direction = "mayor" if anomaly_means[feature] > normal_means[feature] else "menor"
                diff_descriptions.append(f"{feature} {direction} que registros normales")
            
            insights["dbscan_insights"]["anomalies"] = {
                "count": len(anomalies),
                "percentage": anomaly_percentage,
                "distinctive_features": diff_descriptions
            }
            
            anomaly_desc = f"Anomalías DBSCAN ({anomaly_percentage:.1f}% de registros): {', '.join(diff_descriptions)}"
            insights["combined_insights"].append(anomaly_desc)
        
        # Analizar cada cluster normal
        for cluster_id in sorted(list(set(dbscan_df['dbscan_cluster'].unique()) - {-1})):
            cluster_data = dbscan_df[dbscan_df['dbscan_cluster'] == cluster_id]
            cluster_percentage = len(cluster_data) / len(dbscan_df) * 100
            
            insights["dbscan_insights"][f"cluster_{cluster_id}"] = {
                "size": len(cluster_data),
                "percentage": cluster_percentage
            }
    
    # 3. Insights de PCA - Interpretación de componentes principales
    if all(f'PC{i+1}' in pca_df.columns for i in range(2)):
        # Obtener importancia de características para las primeras dos componentes
        if 'feature_importance' in pca_metrics:
            pc1_importance = pca_metrics['feature_importance']['PC1']
            pc2_importance = pca_metrics['feature_importance']['PC2']
            
            # Ordenar características por importancia para cada componente
            pc1_top_features = sorted(pc1_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            pc2_top_features = sorted(pc2_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            
            # Crear descripciones
            pc1_desc = [f"{feature} ({'positivo' if value > 0 else 'negativo'})" for feature, value in pc1_top_features]
            pc2_desc = [f"{feature} ({'positivo' if value > 0 else 'negativo'})" for feature, value in pc2_top_features]
            
            insights["pca_insights"]["principal_components"] = {
                "PC1_explained_variance": pca_metrics['explained_variance_ratio'][0],
                "PC1_top_features": pc1_desc,
                "PC2_explained_variance": pca_metrics['explained_variance_ratio'][1],
                "PC2_top_features": pc2_desc
            }
            
            pc1_explanation = f"PC1 ({pca_metrics['explained_variance_ratio'][0]*100:.1f}% de varianza): Principalmente {', '.join(pc1_desc)}"
            pc2_explanation = f"PC2 ({pca_metrics['explained_variance_ratio'][1]*100:.1f}% de varianza): Principalmente {', '.join(pc2_desc)}"
            
            insights["combined_insights"].append(pc1_explanation)
            insights["combined_insights"].append(pc2_explanation)
    
    # 4. Insights de Isolation Forest - Patrones de anomalías climáticas
    if 'is_anomaly' in isolation_df.columns:
        anomalies = isolation_df[isolation_df['is_anomaly'] == 1]
        normal = isolation_df[isolation_df['is_anomaly'] == 0]
        
        if len(anomalies) > 0 and len(normal) > 0:
            # Comparar valores medios de anomalías vs registros normales
            anomaly_means = anomalies.drop(columns=['is_anomaly', 'anomaly_score', 'Location'] 
                                         if 'Location' in anomalies.columns 
                                         else ['is_anomaly', 'anomaly_score']).mean()
            
            normal_means = normal.drop(columns=['is_anomaly', 'anomaly_score', 'Location']
                                     if 'Location' in normal.columns
                                     else ['is_anomaly', 'anomaly_score']).mean()
            
            # Encontrar las características con mayor diferencia
            differences = (anomaly_means - normal_means).abs().sort_values(ascending=False)
            top_diff_features = differences.head(3)
            
            diff_descriptions = []
            for feature, value in top_diff_features.items():
                direction = "mayor" if anomaly_means[feature] > normal_means[feature] else "menor"
                percentage = abs(anomaly_means[feature] - normal_means[feature]) / normal_means[feature] * 100
                diff_descriptions.append(f"{feature} {direction} ({percentage:.1f}% diferencia)")
            
            insights["anomaly_insights"] = {
                "count": len(anomalies),
                "percentage": len(anomalies) / len(isolation_df) * 100,
                "distinctive_features": diff_descriptions
            }
            
            anomaly_desc = f"Anomalías climáticas ({len(anomalies) / len(isolation_df) * 100:.1f}% de registros): {', '.join(diff_descriptions)}"
            insights["combined_insights"].append(anomaly_desc)
    
    # 5. Insights combinados - Relaciones entre diferentes modelos
    # Comparar clusters de K-means con anomalías de Isolation Forest
    if 'cluster' in kmeans_df.columns and 'is_anomaly' in isolation_df.columns:
        # Unir resultados
        combined_df = kmeans_df.copy()
        combined_df['is_anomaly'] = isolation_df['is_anomaly']
        
        # Ver distribución de anomalías por cluster
        anomalies_by_cluster = combined_df.groupby('cluster')['is_anomaly'].mean() * 100
        
        # Identificar clusters con más anomalías
        high_anomaly_clusters = anomalies_by_cluster[anomalies_by_cluster > anomalies_by_cluster.mean()]
        
        if not high_anomaly_clusters.empty:
            cluster_ids = high_anomaly_clusters.index.tolist()
            desc = f"Clusters con alta concentración de anomalías: {', '.join([str(c) for c in cluster_ids])} " + \
                   f"(promedio de {high_anomaly_clusters.mean():.1f}% de anomalías vs {anomalies_by_cluster.mean():.1f}% global)"
            insights["combined_insights"].append(desc)
    
    logger.info(f"Generados {len(insights['combined_insights'])} insights principales")
    return insights