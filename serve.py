from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import joblib
import pandas as pd
from typing import Dict, Union, Literal
import uvicorn
from enum import Enum
from datetime import datetime
import time
import json
from contextlib import asynccontextmanager
from collections import OrderedDict

#Manejo del ciclo de vida de la aplicación
@asynccontextmanager
async def lifespan(app: FastAPI):
    #inicio (startup)
    load_models()
    yield

app = FastAPI(
    title="API de Predicción de Modelos Entrenados",
    description="Esta API permite realizar predicciones con modelos previamente entrenados.",
    version="1.0",
    lifespan=lifespan
)

MODELS_DIR = "models"
loaded_models = {}

class ModelosEnum(str, Enum):
    LogisticRegression = "LogisticRegression"
    RandomForestClassifier = "RandomForestClassifier"
    SVC = "SVC"
    
class PassengerFeatures(BaseModel):
    PassengerId: int = Field(..., title="Passenger ID", description="Identificador único del pasajero")
    Pclass: int = Field(..., title="Clase", description="Clase en la que viajaba el pasajero (1, 2 o 3)")
    Name: str = Field(..., title="Nombre", description="Nombre completo del pasajero")
    Sex: Literal["male", "female"] = Field(..., title="Sexo", description="Sexo: 'male' o 'female'")
    Age: float = Field(..., title="Edad", description="Edad del pasajero en años")
    SibSp: int = Field(..., title="Hermanos/Pareja a bordo", description="Número de hermanos/esposas a bordo")
    Parch: int = Field(..., title="Padres/Hijos a bordo", description="Número de padres/hijos a bordo")
    Ticket: str = Field(..., title="Boleto", description="Número o código del boleto (puede incluir letras y espacios ej. 'A/5 21171', 'PC 17599', '113803', '373450', '330877',..)")
    Fare: float = Field(..., title="Tarifa", description="Tarifa pagada por el boleto")
    Cabin: str = Field(..., title="Cabina", description="Cabina asignada (puede ser alfanumérica, ej: 'C85', 'C123', 'G6', 'A6', 'C23',..)")
    Embarked: str = Field(..., title="Puerto de embarque", description="C = Cherbourg, Q = Queenstown, S = Southampton (ej. 'C', 'Q', 'S')")

class InputData(BaseModel):
    data: PassengerFeatures

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "PassengerId": 11,
                    "Pclass": 3,
                    "Name": "B",
                    "Sex": "male",
                    "Age": 25,
                    "SibSp": 1,
                    "Parch": 1,
                    "Ticket": "A/5 21171",
                    "Fare": 7.25,
                    "Cabin": "G6",
                    "Embarked": "S"
                }
            }
        }

def load_models():
    """Carga modelos entrenados"""
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".pkl"):
            model_name = file.replace(".pkl", "")
            loaded_models[model_name] = joblib.load(os.path.join(MODELS_DIR, file))
    print("Modelos disponibles:", list(loaded_models.keys()))

@app.get("/modelos", summary="Listar modelos disponibles")
def modelos_disponibles():
    """Devuelve los modelos disponibles"""
    return {"modelos": list(loaded_models.keys())}

@app.post("/predict/{modelo}", summary="Predecir usando un modelo entrenado")
def predecir(modelo: ModelosEnum, input_data: InputData):
    if modelo not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modelo '{modelo}' no encontrado.")
    
    input_dict = input_data.data.dict()
    df = pd.DataFrame([input_dict])
    
    model = loaded_models[modelo]
    
    try:
        expected_cols = model.feature_names_in_
        df = df[expected_cols]
    
        start_time = time.time()
        pred = model.predict(df)
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 2)
        
        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "modelo": modelo,
            "input": input_dict,
            "prediccion": pred.tolist(),
            "latencia_ms": latency_ms
        }
        
        os.makedirs("logs", exist_ok=True)
        with open("logs/inferencia.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")
        
        return {
            "modelo": modelo,
            "entrada": input_dict,
            "prediccion": pred.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al hacer predicción: {str(e)}")
        
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000)
