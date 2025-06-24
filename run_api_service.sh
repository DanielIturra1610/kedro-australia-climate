#!/bin/bash

# Script para desplegar la API del proyecto Australia Climate Analysis
echo "Desplegando servicios de Australia Climate Analysis..."

# Construir y levantar los servicios
docker-compose up --build -d api

echo "Servicios desplegados correctamente."
echo "La API está disponible en http://localhost:8000"
echo "Documentación Swagger UI: http://localhost:8000/docs"

# Instrucciones adicionales
echo -e "\nComandos útiles:"
echo "  - Ver logs de la API: docker-compose logs -f api"
echo "  - Detener servicios: docker-compose down"
echo "  - Ejecutar pipeline Kedro: docker-compose exec kedro kedro run --pipeline=<nombre_pipeline>"
echo "  - Ver documentación de la API: abrir http://localhost:8000/docs en el navegador"
