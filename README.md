# Sistema de ML con Ray, FastAPI y Streamlit

Este proyecto implementa una plataforma completa de Machine Learning usando Ray para computación distribuida, FastAPI como backend y Streamlit como frontend.

## 🏗️ Arquitectura

- **Ray Head**: Nodo principal del clúster Ray
- **Ray Workers**: Nodos trabajadores para procesamiento distribuido
- **FastAPI**: API backend para manejar entrenamientos y predicciones
- **Streamlit**: Dashboard frontend para interacción con el usuario

## 🚀 Inicio Rápido

### Opción 1: Script Automático (Recomendado)
```bash
chmod +x start_system.sh
./start_system.sh
```

### Opción 2: Manual
```bash
# Construir y iniciar todos los servicios
docker-compose up --build

# O iniciar en segundo plano
docker-compose up -d --build
```

## 🔍 Verificación del Sistema

### Verificar estado de los servicios
```bash
docker-compose ps
```

### Diagnóstico de Ray
```bash
docker-compose exec model-api python debug_ray.py
```

### Ver logs de un servicio específico
```bash
docker-compose logs -f [service_name]
# Ejemplos:
docker-compose logs -f ray-head
docker-compose logs -f model-api
docker-compose logs -f dashboard
```

## 🌐 Acceso a los Servicios

- **Streamlit Dashboard**: http://localhost:8501
- **FastAPI API**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8000/docs
- **Ray Dashboard**: http://localhost:8265

## 📊 Uso del Sistema

### 1. Subir Dataset
- Ve al dashboard de Streamlit (http://localhost:8501)
- Sube un archivo CSV o JSON
- Selecciona la columna objetivo
- Elige los modelos a entrenar

### 2. Entrenar Modelos
- Haz clic en "Iniciar Entrenamiento"
- El sistema procesará los datos usando Ray
- Podrás ver el progreso en tiempo real

### 3. Realizar Predicciones
- Una vez entrenados los modelos, aparecerán en la sección de predicción
- Introduce las características en formato JSON
- Obtén predicciones instantáneas

## 🛠️ Solución de Problemas

### Error: "Can't run an actor the server doesn't have a handle for"

Este error indica problemas de conectividad con Ray. Soluciones:

1. **Verificar que Ray esté funcionando**:
   ```bash
   docker-compose exec model-api python debug_ray.py
   ```

2. **Reiniciar el sistema**:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

3. **Verificar logs del ray-head**:
   ```bash
   docker-compose logs ray-head
   ```

### Error de conexión en Streamlit

1. **Verificar que FastAPI esté funcionando**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verificar logs de model-api**:
   ```bash
   docker-compose logs model-api
   ```

### Modelos no aparecen después del entrenamiento

1. **Verificar que el entrenamiento se completó**:
   ```bash
   docker-compose logs ray-head
   ```

2. **Ejecutar diagnóstico**:
   ```bash
   docker-compose exec model-api python debug_ray.py
   ```

3. **Recargar la página de Streamlit**

## 🔧 Configuración

### Variables de Entorno

- `RAY_NAMESPACE`: Namespace de Ray (default: "my_ml_models_namespace")
- `FASTAPI_URL`: URL de FastAPI para Streamlit (default: "http://model-api:8000")

### Modelos Soportados

- LogisticRegression
- RandomForestClassifier
- SVC (Support Vector Classifier)

## 📁 Estructura del Proyecto

```
.
├── docker-compose.yml      # Configuración de servicios
├── Dockerfile             # Imagen Docker
├── requirements.txt       # Dependencias Python
├── train.py              # Script de entrenamiento con Ray
├── serve.py              # API FastAPI
├── dashboard.py          # Dashboard Streamlit
├── debug_ray.py          # Script de diagnóstico
├── start_system.sh       # Script de inicio automático
└── README.md            # Este archivo
```

## 🐛 Debug y Desarrollo

### Ejecutar un servicio individualmente
```bash
# Solo Ray head
docker-compose up ray-head

# Solo FastAPI
docker-compose up model-api

# Solo Streamlit
docker-compose up dashboard
```

### Acceder a un contenedor
```bash
docker-compose exec model-api bash
docker-compose exec ray-head bash
```

### Ver todos los logs
```bash
docker-compose logs -f
```

## 🔄 Actualización

Para actualizar el sistema después de cambios en el código:

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📝 Notas Importantes

1. **Memoria**: El sistema requiere al menos 4GB de RAM disponible
2. **Puertos**: Asegúrate de que los puertos 6379, 8000, 8265, 8501, 10001 estén disponibles
3. **Datos**: Los datasets se procesan en memoria, considera el tamaño de tus datos
4. **Persistencia**: Los modelos entrenados se mantienen mientras el clúster Ray esté activo

## 🆘 Soporte

Si encuentras problemas:

1. Ejecuta el diagnóstico: `docker-compose exec model-api python debug_ray.py`
2. Revisa los logs: `docker-compose logs -f`
3. Reinicia el sistema: `docker-compose down && docker-compose up --build`