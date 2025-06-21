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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@ray.remote
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

@ray.remote
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
            print(f"Modelo recibido: {self.model_name}")
            raise ValueError(f"Modelo '{self.model_name}' no está soportado.")
    
    
    def _build_pipeline(self, X_sample):
        # Identifica columnas numéricas y categóricas
        numeric_cols = X_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()

        # Preprocesamiento para columnas numéricas
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # imputa NaNs con la media
            ('scaler', StandardScaler())
        ])

        # Preprocesamiento para columnas categóricas
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # imputa NaNs con la moda
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Aplicar transformaciones
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        # Pipeline final
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

def main():
    try:
        ray.init()
        print("Ray inicializado correctamente")
        logging.info("Ray inicializado. Recursos: %s", ray.available_resources())
    except Exception as e:
        print(f"Error al iniciar Ray: {str(e)}")
        return
    
    try:
        #Carga de datos
        data_loader = DataLoader.remote()
        X_train, X_test, y_train, y_test = ray.get(data_loader.get_train_test.remote(0.2))
        print("Datos cargados y divididos correctamente")
        logging.info("Datos cargados. Train: %s, Test: %s", X_train.shape, X_test.shape)
        
        #Cargar configuración de modelos
        with open('config/models.json') as f:
            models_config = json.load(f)
            print("Configuraciones de modelos cargadas")
            print(models_config)
        logging.info("%d modelos configurados", len(models_config))
        
        futures = []
        for config in models_config:
            name = config["name"]
            params = config.get("params", {})
            trainer = ModelTrainer.remote(name, params)
            future = trainer.train.remote(X_train, y_train)
            futures.append((name, trainer, future))
            
        trained_models = {}
        for name, trainer, future in futures:
            model = ray.get(future)
            acc = ray.get(trainer.evaluate.remote(model, X_test, y_test))
            trained_models[name] = (model, acc)
            logging.info("Modelo '%s' entrenado. Accuracy: %.4f", name, acc)

        os.makedirs("models", exist_ok=True)
        for name, (model, acc) in trained_models.items():
            joblib.dump(model, f"models/{name}.pkl")
            logging.info("Modelo '%s' guardado en /models", name)
        
    except Exception as e:
        logging.error("Error en main: %s", str(e), exc_info=True)
    finally:
        pass 
        #ray.shutdown()
        #print("Ray apagado correctamente")

if __name__ == "__main__":
    main()