# Proyecto: Australia Climate Risk - Configuración Colaborativa

Este proyecto implementa un flujo de minería de datos usando **Kedro**, **PostgreSQL** y **Power BI** para analizar y visualizar riesgos climáticos en Australia.

---

## 🔧 Requisitos del Entorno

### Software necesario:

- Docker + Docker Compose
- Power BI Desktop
- Git (opcional para clonar el repositorio)

---

## 🚀 Levantar el entorno con Docker

Desde la carpeta base del proyecto, ejecutar:

```bash
docker-compose up --build
```

Esto levanta:

- JupyterLab con Kedro (puerto 8888)
- Kedro Viz (puerto 4141)
- PostgreSQL (puerto 5432)

---

## 🔐 Credenciales de la Base de Datos

- **Usuario**: `kedro_user`
- **Contraseña**: `kedro_pass`
- **Base de datos**: `climate_db`
- **Puerto**: `5432`

---

## 🌐 Acceso desde Power BI o DBeaver

### IP del servidor (WSL):

Para obtener la IP interna:

```bash
wsl ip addr show eth0
```

Busca la línea con `inet`. Ejemplo:

```
inet 172.28.81.208/20
```

### Conexión desde Power BI:

1. Fuente de datos > PostgreSQL
2. Servidor: `172.28.81.208:5432`
3. Base de datos: `climate_db`
4. Modo: Importar o DirectQuery
5. Usuario y contraseña

> Si usas DBeaver, configura de igual forma el host e IP del contenedor.

---

## 🔢 Esquema de Guardado de Datos

Los resultados del modelo de regresión se almacenan en la tabla:

**`regression_model_metrics`** con las siguientes columnas:

- `mse`: error cuadrático medio
- `r2_score`: coeficiente R²
- `timestamp`: marca temporal de ejecución

---

## 🔍 Verificación para Compañeros

### 1. Verifica que Docker esté levantado

```bash
docker ps
```

### 2. Abre Power BI o DBeaver

- Usa IP del paso anterior
- Usuario: `kedro_user`
- DB: `climate_db`

### 3. Carga la tabla `regression_model_metrics`

---

## 📁 Estructura del Proyecto (resumen)

- `src/` - Pipelines Kedro (clasificación, regresión, ingeniería de datos)
- `data/` - Datos en diferentes etapas (raw, model input/output, reporting)
- `conf/base/catalog.yml` - Registra datasets y conexiones
- `docker-compose.yml` - Orquestación de servicios
- `postgres_conf/` - Archivos `postgresql.conf` y `pg_hba.conf` personalizados

---

## 🌐 Recursos Adicionales

- Para reiniciar la DB con cambios en la config:

```bash
docker-compose down -v
```

- Accede a Jupyter:
  [http://localhost:8888](http://localhost:8888)
- Accede a Kedro Viz:
  [http://localhost:4141](http://localhost:4141)

---

## 🌟 Contribuciones

Este entorno está preparado para uso colaborativo. Si clonas este repositorio:

1. Asegúrate de tener Docker
2. Ejecuta `docker-compose up --build`
3. Consulta este README y comienza a trabajar

> Si necesitas ayuda, revisa los notebooks en `notebooks/` o pide acceso a `DBeaver` y Power BI.

---

✅ **Estado actual:** configuración completa, tabla de métricas disponible desde Power BI. Listo para conectar modelos adicionales y dashboards.

---
