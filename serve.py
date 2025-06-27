from fastapi import FastAPI, HTTPException
import ray
from pydantic import BaseModel
import pandas as pd
from typing import Dict, List, Any
import os
from datetime import datetime
import time
from contextlib import asynccontextmanager
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define el namespace que usarán todos los actores detached
RAY_NAMESPACE = "my_ml_models_namespace" 

# Definición de los nombres de los modelos a encontrar
EXPECTED_MODEL_NAMES = ["LogisticRegression", "RandomForestClassifier", "SVC"]

# Modelo Pydantic para validar entrada
class PredictionRequest(BaseModel):
    features: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {} # Almacenará los handles de los actores de los modelos
    
    try:
        logger.info("Iniciando conexión a Ray...")
        # Conexión a Ray en el namespace específico
        ray.init(address="ray://ray-head:10001", ignore_reinit_error=True, namespace=RAY_NAMESPACE)
        logger.info(f"Conexión a Ray exitosa en el namespace: {RAY_NAMESPACE}")

        logger.info("Intentando conectar a los actores de modelos individuales...")
        
        for model_name in EXPECTED_MODEL_NAMES:
            model_actor_handle = None
            max_retries = 10
            retry_delay_seconds = 5
            
            for attempt in range(max_retries):
                try:
                    # Intentar obtener el actor en el namespace específico
                    model_actor_handle = ray.get_actor(model_name, namespace=RAY_NAMESPACE)
                    app.state.models[model_name] = model_actor_handle
                    logger.info(f"Actor '{model_name}' obtenido exitosamente en el intento {attempt + 1} del namespace '{RAY_NAMESPACE}'")
                    break
                except ValueError as e:
                    logger.warning(f"Intento {attempt + 1}/{max_retries}: Actor '{model_name}' no encontrado en el namespace '{RAY_NAMESPACE}'. Reintentando en {retry_delay_seconds} segundos. Error: {e}")
                    time.sleep(retry_delay_seconds)
                except Exception as e:
                    logger.error(f"Error inesperado al intentar obtener actor '{model_name}' en el intento {attempt + 1}/{max_retries}: {e}")
                    time.sleep(retry_delay_seconds)
            
            if model_actor_handle is None:
                logger.error(f"Fallo al obtener el actor '{model_name}' después de {max_retries} intentos. Asegúrate de que 'train.py' se haya ejecutado y el actor esté activo en el namespace '{RAY_NAMESPACE}'.")

        if not app.state.models:
            raise RuntimeError("Ningún actor de modelo pudo ser cargado. Asegúrate de que 'train.py' se haya ejecutado correctamente.")

        logger.info(f"Actores de modelos conectados: {list(app.state.models.keys())}")
        
    except Exception as e:
        logger.error(f"Error inicializando Ray o conectando a actores de modelos: {str(e)}")
        raise RuntimeError(f"Error inicializando Ray o conectando a actores de modelos: {str(e)}")
    
    yield
    
    try:
        ray.shutdown()
        logger.info("Ray desconectado")
    except Exception as e:
        logger.error(f"Error desconectando Ray: {str(e)}")
    

app = FastAPI(lifespan=lifespan)
    
@app.get("/models")
async def list_models():
    """Lista todos los modelos disponibles (actores conectados)"""
    try:
        available_models = list(app.state.models.keys())
        return {
            "available_models": available_models,
            "total_models": len(available_models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{model_name}")
async def predict(model_name: str, request: PredictionRequest):
    """Hace predicciones usando modelos entrenados en Ray"""
    try:
        logger.info(f"Solicitud de predicción para modelo: {model_name}")
        
        if model_name not in app.state.models:
            available_models = list(app.state.models.keys())
            raise HTTPException(
                status_code=404, 
                detail=f"Modelo {model_name} no encontrado o no cargado. Modelos disponibles: {available_models}"
            )
        
        model_actor_handle = app.state.models[model_name]
        prediction_result = ray.get(model_actor_handle.predict.remote(request.features))
        
        result = {
            "model": model_name,
            "prediction": prediction_result["prediction"],
            "input_features": request.features
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
    """Endpoint de salud"""
    try:
        model_count = len(app.state.models) if hasattr(app.state, 'models') else 0
        return {
            "status": "healthy",
            "models_loaded": model_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-metadata/{model_name}")
async def get_model_metadata(model_name: str):
    """Obtiene los metadatos de entrenamiento de un modelo específico."""
    try:
        if model_name not in app.state.models:
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
    """Obtiene los metadatos de entrenamiento de todos los modelos cargados."""
    all_metadata = {}
    for model_name, actor_handle in app.state.models.items():
        try:
            metadata = ray.get(actor_handle.get_metadata.remote())
            all_metadata[model_name] = metadata
        except Exception as e:
            all_metadata[model_name] = {"error": f"No se pudo obtener metadatos: {str(e)}"}
    return all_metadata

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
