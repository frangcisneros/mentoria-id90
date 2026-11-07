# Dockerfile para la API de Detección de Ofertas Hoteleras (TP3)
# Mentoría ID90Travel 2026 · Grupo 02

FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos del proyecto necesarios para el runtime
COPY config.py .
COPY auxiliary_functions.py .
COPY api.py .
COPY test_api.py .
COPY data/destination_with_nearest.csv ./data/
COPY data/sample_data_300k.csv.gz ./data/

# Exponer el puerto de la API FastAPI
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
