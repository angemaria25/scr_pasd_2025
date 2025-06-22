from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import pandas as pd
from typing import Dict, Union

app = FastAPI(
    title="API de Predicción de Modelos Entrenados",
    description="Esta API permite realizar predicciones con modelos previamente entrenados.",
    version="1.0"
)

MODELS_DIR = "models"
loaded_models = {}

class InputData(BaseModel):
    data: Dict[str, Union[str, float, int]]


@app.on_event("startup")
def load_models():
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            loaded_models[model_name] = joblib.load(os.path.join(MODELS_DIR, file))
    print("Modelos disponibles:", list(loaded_models.keys()))

@app.get("/")
def index():
    return {
        "mensaje": "Bienvenido a la API de predicciones. Usa /modelos para ver los modelos disponibles."
    }

@app.get("/modelos")
def modelos_disponibles():
    return {"modelos": list(loaded_models.keys())}

@app.post("/predict/{modelo}")
def predecir(modelo: str, input_data: InputData):
    if modelo not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modelo '{modelo}' no encontrado.")
    try:
        df = pd.DataFrame([input_data.data])
        pred = loaded_models[modelo].predict(df)
        return {
            "modelo": modelo,
            "entrada": input_data.data,
            "prediccion": pred.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al predecir: {str(e)}")