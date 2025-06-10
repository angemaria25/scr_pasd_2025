import ray
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os 

#Inicializar Ray
ray.init(include_dashboard=False, _temp_dir="C:/ray_tmp")

#Cargar dataset
data = load_iris(as_frame=True)
df = data.frame
X = df.drop(columns='target')
y = df['target']

#Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Función remota para entrenar modelo
@ray.remote
def train_model(model_class, params, X_train, X_test, y_train, y_test):
    model = model_class(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

#Configuraciones de modelos a entrenar
model_configs = [
    (RandomForestClassifier, {'n_estimators': 50}),
    (RandomForestClassifier, {'n_estimators': 100}),
    (RandomForestClassifier, {'n_estimators': 150}),
]

#Lanzar entrenamiento en paralelo
futures = [
    train_model.remote(model_class, params, X_train, X_test, y_train, y_test)
    for model_class, params in model_configs
]

#Obtener resultados
results = ray.get(futures)

#Guardar modelos
os.makedirs('../models', exist_ok=True)
for i, (model, acc) in enumerate(results):
    print(f"Modelo {i} - Precisión: {acc}")
    joblib.dump(model, f"../models/model_{i}.joblib")

# Cerrar Ray
ray.shutdown()
