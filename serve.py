from fastapi import FastAPI, HTTPException
import ray
from pydantic import BaseModel, Field
from typing import Dict, List, Any
import os
from datetime import datetime
import time
from contextlib import asynccontextmanager
import uvicorn
import logging
from io import StringIO
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAY_NAMESPACE = "my_ml_models_namespace"
EXPECTED_MODEL_NAMES = ["LogisticRegression", "RandomForestClassifier", "SVC"]

class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(
        example={
            "Pclass": 3,
            "Sex": "female",
            "Age": 20,
            "SibSp": 1,
            "Parch": 0,
            "Fare": 7.50,
            "Embarked": "S"
        },
        description="Diccionario de características de entrada para la predicción."
    )

class TrainRequest(BaseModel):
    data: str  # El dataset como una cadena JSON
    target_column: str
    models_to_train: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    app.state.training_orchestrator = None

    try:
        logger.info("Iniciando conexión a Ray desde FastAPI...")
        ray.init(address="ray://ray-head:10001", ignore_reinit_error=True, namespace=RAY_NAMESPACE)
        logger.info(f"Conexión a Ray exitosa en el namespace: {RAY_NAMESPACE}")

        max_retries = 10
        retry_delay_seconds = 5
        for attempt in range(max_retries):
            try:
                app.state.training_orchestrator = ray.get_actor("training_orchestrator", namespace=RAY_NAMESPACE)
                logger.info(f"Actor 'training_orchestrator' obtenido exitosamente en el intento {attempt + 1}")
                break
            except ValueError as e:
                logger.warning(f"Intento {attempt + 1}/{max_retries}: Actor 'training_orchestrator' no encontrado en el namespace '{RAY_NAMESPACE}'. Reintentando en {retry_delay_seconds} segundos. Error: {e}")
                time.sleep(retry_delay_seconds)
            except Exception as e:
                logger.error(f"Error inesperado al intentar obtener actor 'training_orchestrator' en el intento {attempt + 1}/{max_retries}: {e}")
                time.sleep(retry_delay_seconds)

        if app.state.training_orchestrator is None:
            raise RuntimeError("Fallo al obtener el actor 'training_orchestrator' después de varios intentos.")

        connected_models = []
        for model_name in EXPECTED_MODEL_NAMES:
            try:
                model_actor_handle = ray.get_actor(model_name, namespace=RAY_NAMESPACE)
                app.state.models[model_name] = model_actor_handle
                connected_models.append(model_name)
            except ValueError:
                logger.info(f"Actor '{model_name}' no encontrado en el namespace '{RAY_NAMESPACE}' al inicio. Se espera que sea creado o actualizado por el entrenamiento.")
            except Exception as e:
                logger.error(f"Error al intentar obtener actor '{model_name}' durante el inicio: {e}")

        logger.info(f"Actores de modelos conectados inicialmente: {connected_models}")

    except Exception as e:
        logger.error(f"Error inicializando Ray o conectando a actores de modelos: {str(e)}")
        raise RuntimeError(f"Error inicializando Ray o conectando a actores de modelos: {str(e)}")

    yield

    try:
        ray.shutdown()
        logger.info("Ray desconectado")
    except Exception as e:
        logger.error(f"Error desconectando Ray: {str(e)}")


app = FastAPI(lifespan=lifespan,
                title="API de Predicción y Entrenamiento de Modelos",
                description="Esta API permite realizar predicciones y desencadenar entrenamientos utilizando modelos de Machine Learning desplegados como actores en un clúster de Ray. Proporciona endpoints para listar modelos disponibles, realizar predicciones y verificar el estado de salud y de entrenamiento.")

@app.get("/models")
async def list_models():
    """
    Lista todos los modelos disponibles (actores conectados) y sus metadatos.
    """
    available_models_info = []
    try:
        current_active_models = {}
        for model_name in EXPECTED_MODEL_NAMES:
            try:
                actor_handle = ray.get_actor(model_name, namespace=RAY_NAMESPACE)
                current_active_models[model_name] = actor_handle
            except ValueError:
                pass

        app.state.models = current_active_models

        for model_name, actor_handle in app.state.models.items():
            try:
                metadata = ray.get(actor_handle.get_metadata.remote())
                available_models_info.append({
                    "name": model_name,
                    "status": "loaded",
                    "metrics": metadata.get("metrics"),
                    "training_date": metadata.get("training_date")
                })
            except Exception as e:
                available_models_info.append({
                    "name": model_name,
                    "status": f"error: {str(e)}",
                    "metrics": None,
                    "training_date": None
                })
        return {
            "available_models": [m["name"] for m in available_models_info],
            "models_details": available_models_info,
            "total_models": len(available_models_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar modelos: {str(e)}")


@app.post("/predict/{model_name}")
async def predict(model_name: str, request: PredictionRequest):
    """Hace predicciones usando modelos entrenados en Ray"""
    try:
        logger.info(f"Solicitud de predicción para modelo: {model_name}")

        if model_name not in app.state.models:
            try:
                model_actor_handle = ray.get_actor(model_name, namespace=RAY_NAMESPACE)
                app.state.models[model_name] = model_actor_handle
                logger.info(f"Modelo '{model_name}' encontrado y añadido al estado en tiempo de ejecución.")
            except ValueError:
                available_models = list(app.state.models.keys())
                raise HTTPException(
                    status_code=404,
                    detail=f"Modelo {model_name} no encontrado o no cargado. Modelos disponibles: {available_models}"
                )

        model_actor_handle = app.state.models[model_name]

        start_time = time.perf_counter()
        prediction_result = ray.get(model_actor_handle.predict.remote(request.features))
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        result = {
            "model": model_name,
            "prediction": prediction_result["prediction"],
            "input_features": request.features,
            "latency_ms": latency_ms
        }

        if prediction_result["probabilities"] is not None:
            result["probabilities"] = prediction_result["probabilities"]

        logger.info(f"Predicción exitosa para modelo {model_name}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción para modelo {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de salud de la API."""
    try:
        model_count = len(app.state.models) if hasattr(app.state, 'models') else 0
        orchestrator_status = "unavailable"
        if app.state.training_orchestrator:
            try:
                _ = ray.get(app.state.training_orchestrator.get_training_status.remote("test"))
                orchestrator_status = "available"
            except Exception as e:
                orchestrator_status = f"error: {str(e)}"

        return {
            "status": "healthy",
            "models_loaded": model_count,
            "training_orchestrator_status": orchestrator_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-metadata/{model_name}")
async def get_model_metadata(model_name: str):
    """Obtiene los metadatos de entrenamiento de un modelo específico."""
    try:
        if model_name not in app.state.models:
            try:
                actor_handle = ray.get_actor(model_name, namespace=RAY_NAMESPACE)
                app.state.models[model_name] = actor_handle
            except ValueError:
                raise HTTPException(status_code=404, detail=f"Modelo {model_name} no encontrado.")

        model_actor_handle = app.state.models[model_name]
        metadata = ray.get(model_actor_handle.get_metadata.remote())
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al obtener metadatos para {model_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/all-models-metadata")
async def get_all_models_metadata():
    """
    Obtiene los metadatos de entrenamiento de todos los modelos cargados.
    """
    all_metadata = {}
    connected_models_handles = {}

    for model_name in EXPECTED_MODEL_NAMES:
        try:
            actor_handle = ray.get_actor(model_name, namespace=RAY_NAMESPACE)
            connected_models_handles[model_name] = actor_handle
        except ValueError:
            pass
        except Exception as e:
            logger.error(f"Error al intentar obtener actor '{model_name}' para metadatos: {e}")

    app.state.models = connected_models_handles

    for model_name, actor_handle in app.state.models.items():
        try:
            metadata = ray.get(actor_handle.get_metadata.remote())
            all_metadata[model_name] = metadata
        except Exception as e:
            all_metadata[model_name] = {"error": f"No se pudo obtener metadatos: {str(e)}"}
            logger.error(f"Error al obtener metadatos de {model_name}: {e}")
    return all_metadata


@app.post("/train-models")
async def train_models(request: TrainRequest):
    """
    Desencadena el entrenamiento de modelos con el dataset (enviado como JSON)
    y la columna objetivo especificados utilizando el TrainingOrchestrator de Ray.
    """
    if not app.state.training_orchestrator:
        raise HTTPException(status_code=503, detail="El orquestador de entrenamiento de Ray no está disponible.")

    try:
        # Convertir la cadena JSON a un DataFrame de Pandas
        df_to_process = pd.read_json(StringIO(request.data), orient='records')
        logger.info(f"Dataset recibido y convertido a DataFrame en FastAPI. Shape: {df_to_process.shape}")

        # Poner el DataFrame en el Object Store de Ray
        data_ref = ray.put(df_to_process)
        logger.info(f"Dataset colocado en el Object Store de Ray con ObjectRef: {data_ref}")

        # Llamar al actor TrainingOrchestrator para iniciar el entrenamiento, pasando el ObjectRef
        training_response = ray.get(
            app.state.training_orchestrator.start_training.remote(
                data_ref, request.target_column, request.models_to_train
            )
        )
        return training_response

    except Exception as e:
        logger.error(f"Error al iniciar el entrenamiento a través del orquestador: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al iniciar el entrenamiento: {str(e)}")

@app.get("/training-status/{task_id}")
async def get_training_status(task_id: str):
    """
    Obtiene el estado de una tarea de entrenamiento específica por su ID.
    """
    if not app.state.training_orchestrator:
        raise HTTPException(status_code=503, detail="El orquestador de entrenamiento de Ray no está disponible.")
    try:
        status_info = ray.get(app.state.training_orchestrator.get_training_status.remote(task_id))
        return status_info
    except Exception as e:
        logger.error(f"Error al obtener el estado de la tarea de entrenamiento {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al obtener estado de entrenamiento: {str(e)}")