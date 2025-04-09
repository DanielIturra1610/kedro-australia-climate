# Usa una imagen ligera y moderna
FROM python:3.10-slim

# Variables de entorno recomendadas
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Establece directorio de trabajo
WORKDIR /home/kedro

# Instala dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia requerimientos y los instala
COPY src/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instala Kedro, Jupyter y Kedro Viz
RUN pip install kedro==0.18.10 jupyterlab kedro-viz

# Copia el contenido completo del proyecto
COPY . .

# Exposición del puerto de Jupyter
EXPOSE 8888

# Comando por defecto
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
