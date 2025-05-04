import pandas as pd
import numpy as np
import logging
import joblib   
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def imputar_min_max_temp(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[imputar_min_max_temp] Shape inicial: %s", df.shape)
    logger.info("[imputar_min_max_temp] Nulos antes:\n%s", df[['MinTemp', 'MaxTemp']].isnull().sum())

    df['MinTemp'] = df['MinTemp'].fillna(df['MinTemp'].median())
    df['MaxTemp'] = df['MaxTemp'].fillna(df['MaxTemp'].median())

    logger.info("[imputar_min_max_temp] Nulos después:\n%s", df[['MinTemp', 'MaxTemp']].isnull().sum())
    return df


def imputar_rain_today(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[imputar_rain_today] Valores únicos antes: %s", df['RainToday'].unique())
    df['RainToday'] = df['RainToday'].fillna(df['RainToday'].mode()[0])
    logger.info("[imputar_rain_today] Nulos después: %s", df['RainToday'].isnull().sum())
    return df


def convertir_fecha(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[convertir_fecha] Tipo de 'Date' antes: %s", df['Date'].dtype)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    logger.info("[convertir_fecha] Tipo de 'Date' después: %s", df['Date'].dtype)
    return df


def extraer_caracteristicas_temporales(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("[extraer_caracteristicas_temporales] Agregando columnas temporales...")

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter

    def get_season(month):
        if month in [12, 1, 2]:
            return 'Verano'
        elif month in [3, 4, 5]:
            return 'Otoño'
        elif month in [6, 7, 8]:
            return 'Invierno'
        else:
            return 'Primavera'

    df['Season'] = df['Month'].apply(get_season)
    logger.info("[extraer_caracteristicas_temporales] Temporadas únicas: %s", df['Season'].unique())
    return df


def detectar_outliers_rainfall(df: pd.DataFrame) -> pd.DataFrame:
    Q1 = df['Rainfall'].quantile(0.25)
    Q3 = df['Rainfall'].quantile(0.75)
    IQR = Q3 - Q1

    logger.info("[detectar_outliers_rainfall] Q1: %.2f, Q3: %.2f, IQR: %.2f", Q1, Q3, IQR)

    df['Rainfall_Outlier'] = ((df['Rainfall'] < Q1 - 1.5 * IQR) | (df['Rainfall'] > Q3 + 1.5 * IQR)).astype(int)

    outlier_count = df['Rainfall_Outlier'].sum()
    logger.info("[detectar_outliers_rainfall] Outliers detectados: %d", outlier_count)
    return df


def predecir_riesgo_climatico(modelo, datos) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info("[PREDICCIÓN] Ejecutando inferencia de riesgo climático...")

    datos = datos.copy()

    # Asegurarse de que todas las columnas esperadas estén presentes
    for col in modelo.feature_names_in_:
        if col not in datos.columns:
            datos[col] = 0

    datos = datos[modelo.feature_names_in_]
    predicciones = modelo.predict(datos)

    logger.info("[PREDICCIÓN] Predicciones completadas.")
    datos['PredictedRainTomorrow'] = predicciones
    return datos



def train_climate_risk_classifier(df):
    import logging
    logger = logging.getLogger(__name__)

    logger.info("[CLASIFICADOR] Filtrando columnas relevantes...")
    features = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'RainToday', 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']
    target = 'RainTomorrow'

    df = df.dropna(subset=features + [target])
    df = df.copy()

    # Convertir RainToday y RainTomorrow a 1/0
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

    # One-hot para Location
    df = pd.get_dummies(df, columns=['Location'], drop_first=True)

    location_dummies = [col for col in df.columns if col.startswith('Location_')]
    X = df[[col for col in features if col != 'Location'] + location_dummies]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("[CLASIFICADOR] Entrenando modelo RandomForest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    logger.info("[CLASIFICADOR] Resultados del modelo:")
    logger.info("\n" + classification_report(y_test, y_pred))

    # Guardar modelo entrenado para predicción futura
    joblib.dump(clf, "data/06_models/climate_risk_classifier.pkl")

    return clf, pd.DataFrame({"predictions": y_pred.tolist()})


def infer_climate_risk(modelo, df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info("[INFERENCIA] Ejecutando inferencia usando el modelo en memoria...")

    df = df.copy()
    expected_features = modelo.feature_names_in_
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # agregamos columnas faltantes como 0

    df = df[expected_features]
    predictions = modelo.predict(df)

    logger.info("[INFERENCIA] Predicciones realizadas: %s", np.unique(predictions, return_counts=True))
    df['PredictedRainTomorrow'] = predictions
    return df


def preparar_climate_inference_input(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    logger.info("[INFERENCIA] Preparando datos para inferencia...")

    df = df.copy()

    # Eliminar filas con nulos en las columnas requeridas
    required_columns = ['Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'RainToday',
                        'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter']
    df = df.dropna(subset=required_columns)

    # Codificar RainToday
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})

    # One-hot encoding para Location (igual que en entrenamiento)
    df = pd.get_dummies(df, columns=['Location'], drop_first=True)

    # Ordenar columnas: primero las que no son Location_ y luego las que sí
    location_cols = [col for col in df.columns if col.startswith('Location_')]
    ordered_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'RainToday',
                    'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter'] + location_cols

    df = df[ordered_cols]

    logger.info("[INFERENCIA] Dataset listo para inferencia con shape: %s", df.shape)
    return df


def calcular_indice_riesgo_climatico(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el índice de riesgo climático por ciudad.

    Args:
        df (pd.DataFrame): DataFrame con las predicciones climáticas. 
                           Debe incluir columnas de location dummies y 'PredictedRainTomorrow'.

    Returns:
        pd.DataFrame: Índice de riesgo por ciudad.
    """
    location_cols = [col for col in df.columns if col.startswith("Location_")]

    # Transformamos dummies a nombres de ciudad
    melted = df[location_cols + ["PredictedRainTomorrow"]].copy()
    melted = melted.melt(id_vars="PredictedRainTomorrow", var_name="City", value_name="is_city")
    melted = melted[melted["is_city"] == True].drop(columns="is_city")

    # Extraemos nombre limpio de la ciudad
    melted["City"] = melted["City"].str.replace("Location_", "")

    # Cálculo del índice de riesgo
    risk_index = (
        melted.groupby("City")["PredictedRainTomorrow"]
        .mean()
        .reset_index()
        .rename(columns={"PredictedRainTomorrow": "ClimateRiskIndex"})
    )

    return risk_index

def train_regression_model(df: pd.DataFrame, params: dict) -> tuple:
    """Entrena un modelo de regresión basado en las características entregadas.

    Args:
        df: DataFrame de entrada con características y variable objetivo.
        params: Diccionario de parámetros definidos en parameters.yml.

    Returns:
        Tuple: (modelo entrenado, métricas del modelo en DataFrame)
    """
    # Definimos variables predictoras y target
    features = df.drop(columns=["Rainfall"])
    target = df["Rainfall"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=params["random_state"]
    )

    # Inicializar el modelo de regresión
    if params["model_type"] == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
            n_jobs=params["n_jobs"]
        )
    else:
        raise ValueError(f"Modelo de regresión {params['model_type']} no soportado todavía.")

    # Entrenar modelo
    model.fit(X_train, y_train)

    # Evaluar modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    metrics = pd.DataFrame({
        "mse": [mse],
        "rmse": [rmse],
    })

    return model, metrics

def save_model(model, output_path: str) -> None:
    """Guarda el modelo entrenado en un archivo .pkl."""
    joblib.dump(model, output_path)
