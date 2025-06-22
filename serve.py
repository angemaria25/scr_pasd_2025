from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import uvicorn
from typing import List, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

app = FastAPI(
    title="API de Predicción de Modelos Entrenados",
    description="Permite hacer predicciones y reentrenar modelos sobre datos nuevos.",
    version="1.0"
)

MODELS_DIR = "models"
loaded_models: Dict[str, object] = {}
model_accuracy: Dict[str, float] = {}

class InputData(BaseModel):
    data: Dict[str, Union[str, float, int]]

class TrainData(BaseModel):
    data: List[Dict[str, Union[str, float, int]]]

@app.on_event("startup")
def cargar_modelos():
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".pkl"):
            nombre = file.replace(".pkl", "")
            path = os.path.join(MODELS_DIR, file)
            loaded_models[nombre] = joblib.load(path)
    print("Modelos cargados:", list(loaded_models.keys()))

@app.get("/")
def raiz():
    return {"mensaje": "Bienvenido a la API de Predicción de Datos de Modelos Entrenados"}

@app.get("/modelos")
def listar_modelos():
    if not loaded_models:
        return {"modelos": [], "mensaje": "No hay modelos cargados actualmente."}
    return {"modelos_disponibles": list(loaded_models.keys())}

@app.post("/predict/{modelo}")
def predecir(modelo: str, input_data: InputData):
    if modelo not in loaded_models:
        raise HTTPException(status_code=404, detail=f"Modelo '{modelo}' no encontrado.")
    
    df = pd.DataFrame([input_data.data])
    try:
        pred = loaded_models[modelo].predict(df)
        return {"modelo": modelo, "prediccion": pred.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reentrenar/{modelo}")
def reentrenar_modelo(modelo: str, datos: TrainData):
    if modelo not in ["RandomForestClassifier", "LogisticRegression", "SVC"]:
        raise HTTPException(status_code=400, detail=f"Modelo '{modelo}' no es compatible para reentrenamiento directo.")

    df = pd.DataFrame(datos.data)
    if "target" not in df.columns:
        raise HTTPException(status_code=400, detail="Los datos deben incluir una columna 'target'.")

    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if modelo == "RandomForestClassifier":
        model = RandomForestClassifier()
    elif modelo == "LogisticRegression":
        model = LogisticRegression()
    else:
        model = SVC()

    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    ruta = os.path.join(MODELS_DIR, f"{modelo}.pkl")
    joblib.dump(model, ruta)
    loaded_models[modelo] = model
    model_accuracy[modelo] = acc

    return {
        "mensaje": f"Modelo '{modelo}' reentrenado exitosamente.",
        "accuracy_test": round(acc, 4)
    }
