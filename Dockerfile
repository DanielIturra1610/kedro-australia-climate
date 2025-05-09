# ───────── 1. Imagen base ─────────
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /home/kedro

# ───────── 2. Paquetes de sistema (con reintento) ─────────
RUN set -eux; \
    # primer intento
    apt-get update && \
    apt-get -o Acquire::Retries=3 install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/* \
    # si falla, segundo intento tras 20 s
    || ( \
    echo "⏳ 1er intento falló; reintento en 20 s…" && \
    sleep 20 && \
    apt-get update --fix-missing && \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/* )

# ───────── 3. Dependencias Python ─────────
COPY src/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ───────── 4. Copia del proyecto ─────────
COPY . .

# ───────── 5. Exponer Jupyter (opcional) ─────────
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
