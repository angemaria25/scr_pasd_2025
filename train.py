import ray 
import json
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import time
from ray.exceptions import ActorDiedError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@ray.remote(max_restarts=3, max_task_retries=3)
class DataLoader:
    def __init__(self, data_dir="data"):
        """Carga automática de archivos CSV o JSON"""
        self.data = self._load_data(data_dir)
    
    def _load_data(self, data_dir):
        """Busca y carga el primer archivo CSV o JSON encontrado"""
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                return pd.read_csv(os.path.join(data_dir, file))
            elif file.endswith('.json'):
                return pd.read_json(os.path.join(data_dir, file))
        raise FileNotFoundError("No se encontraron archivos CSV o JSON en /data")

    def get_train_test(self, test_size=0.2):
        """Prepara los datos para entrenamiento"""
        if 'target' not in self.data.columns:
            raise ValueError("El dataset debe contener columna 'target'")
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        return train_test_split(X, y, test_size=test_size, random_state=42)

@ray.remote(max_restarts=3, max_task_retries=3)
class ModelTrainer:
    def __init__(self, model_name, model_params):
        self.model_name = model_name
        self.model_params = model_params
        self.model = self._build_model()
    
    def _build_model(self):
        """Inicializa el modelo según nombre y parámetros"""
        if self.model_name == "LogisticRegression":
            return LogisticRegression(**self.model_params)
        elif self.model_name == "RandomForestClassifier":
            return RandomForestClassifier(**self.model_params)
        elif self.model_name == "SVC":
            return SVC(**self.model_params)
        else:
            raise ValueError(f"Modelo '{self.model_name}' no está soportado.")
    
    def _build_pipeline(self, X_sample):
        # Identifica columnas numéricas y categóricas
        numeric_cols = X_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # imputa NaNs con la media
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # imputa NaNs con la moda
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])

    def train(self, X_train, y_train):
        """Entrena el modelo"""
        pipeline = self._build_pipeline(X_train)
        pipeline.fit(X_train, y_train)
        return pipeline
    
    def evaluate(self, model, X_test, y_test):
        """Evalúa el modelo entrenado"""
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return acc
    

@ray.remote 
def safe_train(name, params, X_train, y_train, X_test, y_test, retries=3):
    """Recrear actor manualmente en otro nodo"""
    for attempt in range(retries):
        try:
            trainer = ModelTrainer.remote(name, params)
            model = ray.get(trainer.train.remote(X_train, y_train))
            acc = ray.get(trainer.evaluate.remote(model, X_test, y_test))
            return name, model, acc
        except ActorDiedError:
            print(f"[{name}] Intento {attempt + 1} fallido por ActorDiedError. Reintentando...")
            time.sleep(2)
    return name, None, None


def main():
    try:
        ray.init()
        print("Ray inicializado correctamente")
        logging.info("Ray inicializado. Recursos: %s", ray.available_resources())
    except Exception as e:
        print(f"Error al iniciar Ray: {str(e)}")
        return
    
    try:
        data_loader = DataLoader.remote()
        X_train, X_test, y_train, y_test = ray.get(data_loader.get_train_test.remote(0.2))
        print("Datos cargados y divididos correctamente")
        logging.info("Datos cargados. Train: %s, Test: %s", X_train.shape, X_test.shape)
        
        with open('config/models.json') as f:
            models_config = json.load(f)
            print("Configuraciones de modelos cargadas")
            print(models_config)
        logging.info("%d modelos configurados", len(models_config))
        
        #Lanzar todos los entrenamientos en paralelo
        futures = []
        for config in models_config:
            name = config["name"]
            params = config.get("params", {})
            future = safe_train.remote(name, params, X_train, y_train, X_test, y_test)
            futures.append(future)
                
        #Esperar a que todos terminen
        results = ray.get(futures)
        
        #Guardar modelos entrenados
        trained_models = {}
        os.makedirs("models", exist_ok=True)
        
        for name, model, acc in results:
            if model is not None:
                trained_models[name] = (model, acc)
                joblib.dump(model, f"models/{name}.pkl")
                logging.info("Modelo '%s' entrenado y guardado. Accuracy: %.4f", name, acc)
            else:
                logging.warning("El modelo '%s' falló después de varios intentos.", name)
                
                
    except Exception as e:
        logging.error("Error en main: %s", str(e), exc_info=True)
    finally:
        ray.shutdown()
        
if __name__ == "__main__":
    main()