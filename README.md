# 🌡️ Australia Climate Risk Analysis

<div align="center">
  
![Kedro](https://img.shields.io/badge/Kedro-0.18.14-blue)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.0-336791)
![Python](https://img.shields.io/badge/Python-3.10-yellow)
![Power BI](https://img.shields.io/badge/Power_BI-Latest-F2C811)
![Docker](https://img.shields.io/badge/Docker-Latest-2496ED)

</div>

## 📋 Descripción

Un sistema de análisis y predicción de riesgos climáticos en Australia basado en datos históricos meteorológicos. El proyecto implementa una plataforma de ciencia de datos completa que incluye:

- **Pipelines de procesamiento de datos** con Kedro
- **Modelos predictivos** (SVM, Naive Bayes, Regresión)
- **Almacenamiento y gestión de métricas** en PostgreSQL
- **Visualización interactiva** con Power BI

El objetivo principal es predecir patrones climáticos y evaluar los factores de riesgo asociados a eventos climáticos extremos en diferentes ubicaciones de Australia.

## 🧠 Modelos Implementados

### 🔹 Predicción de Lluvia con Naive Bayes
Implementa un clasificador Gaussiano para predecir si lloverá al día siguiente basado en variables meteorológicas actuales.

### 🔹 Regresión con SVM Optimizado
Modelo de regresión SVM para variables climáticas continuas con optimizaciones especiales:
- Kernel lineal para mejor rendimiento
- Parámetro C reducido para acelerar convergencia
- Muestreo inteligente para datasets grandes

### 🔹 Modelos Adicionales
- Random Forest
- Regresión Múltiple
- Modelos de clasificación para análisis de riesgo

## 🛠️ Requisitos del Entorno

### Software necesario:

- Docker + Docker Compose
- Power BI Desktop
- Git (opcional para clonar el repositorio)

## 🚀 Inicio Rápido

### 1. Levantar el entorno con Docker

```bash
docker-compose up --build
```

Esto despliega:
- JupyterLab con Kedro (puerto 8888)
- Kedro Viz (puerto 4141)
- PostgreSQL (puerto 5432)

### 2. Acceder a las herramientas

- Jupyter Lab: [http://localhost:8888](http://localhost:8888)
- Kedro Viz: [http://localhost:4141](http://localhost:4141)
- Base de datos PostgreSQL (puerto 5432)

## 💾 Base de Datos

### Tablas Principales

| Tabla | Descripción |
|-------|-------------|
| `public.climate_risk` | Índice de riesgo climático |
| `public.metadata.runs` | Metadatos de ejecuciones |
| `public.ml_metrics.classification` | Métricas de clasificación (Naive Bayes) |
| `public.ml_metrics.regression` | Métricas de regresión (SVM, Random Forest) |

> **Nota**: Los nombres de tablas contienen puntos, lo que requiere usar comillas dobles en consultas SQL:
> ```sql
> SELECT * FROM "public"."ml_metrics.regression"
> ```

## 🔌 Conexión desde Power BI

### 1. Obtener la IP del servidor (WSL)

```bash
wsl ip addr show eth0
```

Busca la línea con `inet`. Ejemplo:
```
inet 172.28.81.208/20
```

## 📂 Estructura del Proyecto

```
kedro_aus_climate/
│
├── conf/                    # Configuración del proyecto
│   └── base/                
│       └── catalog.yml      # Configuración de datasets y conexiones
│
├── data/                    # Datos en diferentes etapas del pipeline
│
├── docs/                    # Documentación
│
├── notebooks/               # Jupyter notebooks
│
├── src/                     # Código fuente
│   └── australia_climate_analysis/
│       └── pipelines/       # Pipelines de Kedro
│           ├── data_engineering/
│           ├── naive_bayes_forecast/
│           ├── regression_model/
│           └── regression_svm/
│
├── docker-compose.yml       # Configuración de servicios Docker
└── postgres_conf/           # Configuración personalizada de PostgreSQL
```

## 🔧 Comandos Útiles

```bash
# Reiniciar toda la infraestructura
docker-compose down -v && docker-compose up --build

# Ver logs de los contenedores
docker-compose logs -f

# Ejecutar pipeline específico
docker exec -it kedro_container kedro run --pipeline=regression_svm
```

## 🤝 Contribuciones

Este entorno está preparado para uso colaborativo. Si clonas este repositorio:

1. Asegúrate de tener Docker instalado
2. Ejecuta `docker-compose up --build`
3. Consulta la documentación en `docs/` y los notebooks en `notebooks/`

## 📊 Estado Actual

- ✅ Configuración completa 
- ✅ Modelos Naive Bayes y SVM implementados
- ✅ Integración con PostgreSQL funcional
- ✅ Dashboard de Power BI disponible
- ✅ Monitoreo de métricas para modelos de ML

---

<div align="center">
<sub>Desarrollado con ❤️ para el análisis de riesgos climáticos en Australia</sub>
</div>
