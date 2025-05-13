from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    cargar_datos,
    mostrar_valores_nulos,
    imputar_numericas,
    imputar_categoricas,
    convertir_fecha,
    outliers_rainfall,
    agregar_temporales
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=cargar_datos,
            inputs="raw_weather_data",
            outputs="df_cargado",
            name="cargar_datos_node"
        ),
        node(
            func=mostrar_valores_nulos,
            inputs="df_cargado",
            outputs="df_nulos",
            name="valores_nulos_node"
        ),
        node(
            func=imputar_numericas,
            inputs="df_nulos",
            outputs="df_imputado_num",
            name="imputacion_numericas_node"
        ),
        node(
            func=imputar_categoricas,
            inputs="df_imputado_num",
            outputs="df_imputado_cat",
            name="imputacion_categoricas_node"
        ),
        node(
            func=convertir_fecha,
            inputs="df_imputado_cat",
            outputs="df_fecha",
            name="convertir_fecha_data_eng_node"
        ),
        node(
            func=outliers_rainfall,
            inputs="df_fecha",
            outputs="df_outliers",
            name="outliers_node"
        ),
        node(
            func=agregar_temporales,
            inputs="df_outliers",
            outputs="df_final",
            name="temporales_node"
        )
    ])
