# Usa una imagen ligera de Python como base
FROM python:3.10-slim

# Establece un directorio de trabajo dentro del contenedor
WORKDIR /home/kedro

# Instala las dependencias del sistema necesarias para Kedro o operaciones con Git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copia el archivo requirements.txt desde el host al contenedor
COPY src/requirements.txt /home/kedro/requirements.txt

# Instala las dependencias desde el archivo requirements.txt
RUN pip install --no-cache-dir -r /home/kedro/requirements.txt

# Instala Kedro y Jupyter
RUN pip install kedro==0.18.10 jupyter

# Por defecto, inicia una sesión bash
CMD ["/bin/bash"]
