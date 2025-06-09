from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI()
model = joblib.load("/tmp/model.pkl")  # Se monta en Docker

class InferenceRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(req: InferenceRequest):
    x = np.array(req.features).reshape(1, -1)
    pred = model.predict(x)[0]
    return {"prediction": int(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
