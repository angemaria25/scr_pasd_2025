from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import joblib
import pandas as pd
from typing import Dict, Union
import uvicorn
from enum import Enum

app = FastAPI(
    title="API de Predicción de Modelos Entrenados",
    description="Esta API permite realizar predicciones con modelos previamente entrenados.",
    version="1.0"
)

MODELS_DIR = "models"
loaded_models = {}

class ModelosEnum(str, Enum):
    LogisticRegression = "LogisticRegression"
    RandomForestClassifier = "RandomForestClassifier"
    SVC = "SVC"

class InputData(BaseModel):
    """Ejemplo para rellenar el form"""
    data: Dict[str, Union[str, float, int]]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "Pclass": 3,
                    "Name": "A",
                    "Sex": "male",
                    "Age": 22,
                    "SibSp": 1,
                    "Parch": 0,
                    "Ticket": 5,
                    "Fare": 7.25,
                    "Cabin": 6,
                    "Embarked": "S"
                }
            }
        }
    

@app.on_event("startup")
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
    
    df = pd.DataFrame([input_data.data])
    model = loaded_models[modelo]
    
    #validar columnas
    try:
        expected_cols = model.feature_names_in_
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Faltan columnas requeridas: {missing}")
        df = df[expected_cols]
    except AttributeError:
        raise HTTPException(status_code=500, detail="El modelo cargado no contiene metainformación de entrada.")
        
    try:
        pred = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al hacer predicción: {str(e)}")
    
    return {
        "modelo": modelo,
        "entrada": input_data.data,
        "prediccion": pred.tolist()
    }
    
if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000)
