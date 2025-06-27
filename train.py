import ray 
import json
import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import time
from datetime import datetime
from ray.exceptions import ActorDiedError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODELS_CONFIG = [
    {
        "name": "LogisticRegression",
        "params": {
            "solver": "liblinear",
            "random_state": 42
        }
    },
    {
        "name": "RandomForestClassifier",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "random_state": 42
        }
    },
    {
        "name": "SVC",
        "params": {
            "kernel": "rbf",
            "probability": True, 
            "random_state": 42
        }
    }
]

# Actor de Ray para cada Modelo 
@ray.remote(max_restarts=-1, max_task_retries=-1) 
class ModelServiceActor:
    def __init__(self, model_name: str, trained_pipeline: Pipeline, metadata: dict):
        self.model_name = model_name
        self.pipeline = trained_pipeline
        self.metadata = metadata
        logger.info(f"ModelServiceActor '{self.model_name}': Inicializado con pipeline y metadatos.")

    def predict(self, features: dict):
        """
        Realiza una predicción utilizando el pipeline almacenado.
        """
        input_df = pd.DataFrame([features])
        prediction = self.pipeline.predict(input_df)
        
        probabilities = None
        try:
            if hasattr(self.pipeline, 'predict_proba'):
                probabilities = self.pipeline.predict_proba(input_df).tolist()
        except Exception:
            pass

        return {
            "prediction": prediction.tolist(),
            "probabilities": probabilities
        }

    def get_metadata(self):
        """
        Devuelve los metadatos del modelo.
        """
        return self.metadata

    def get_model_name(self):
        """
        Devuelve el nombre del modelo.
        """
        return self.model_name

@ray.remote(max_restarts=3, max_task_retries=3)
class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        logger.info(f"DataLoader: Inicializado con data_dir: {self.data_dir}")
    
    def _load_data(self):
        """Busca y carga el primer archivo CSV o JSON encontrado en self.data_dir."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"DataLoader: El directorio de datos '{self.data_dir}' no existe.")

        for file in os.listdir(self.data_dir):
            full_data_path = os.path.join(self.data_dir, file)
            if file.endswith('.csv'):
                logger.info(f"DataLoader: Cargando CSV desde: {full_data_path}")
                return pd.read_csv(full_data_path)
            elif file.endswith('.json'):
                logger.info(f"DataLoader: Cargando JSON desde: {full_data_path}")
                return pd.read_json(full_data_path)
        raise FileNotFoundError(f"DataLoader: No se encontraron archivos CSV o JSON en {self.data_dir}")

    def get_train_test(self, test_size=0.2, target_column='Survived'):
        """Prepara los datos para entrenamiento."""
        data = self._load_data()
        
        if target_column not in data.columns:
            raise ValueError(f"DataLoader: El dataset debe contener la columna objetivo '{target_column}'. Columnas disponibles: {data.columns.tolist()}")
        
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        logger.info(f"DataLoader: Datos cargados y divididos. X shape: {X.shape}, y shape: {y.shape}")
        
        return ray.put(X), ray.put(y), test_size

@ray.remote(max_restarts=3, max_task_retries=3)
class ModelTrainer:
    def __init__(self, model_name, model_params):
        self.model_name = model_name
        self.model_params = model_params
        self.model = self._build_model()
        logger.info(f"ModelTrainer ({self.model_name}): Inicializado.")
    
    def _build_model(self):
        """Inicializa el modelo según nombre y parámetros."""
        if self.model_name == "LogisticRegression":
            return LogisticRegression(**self.model_params)
        elif self.model_name == "RandomForestClassifier":
            return RandomForestClassifier(**self.model_params)
        elif self.model_name == "SVC":
            return SVC(**self.model_params)
        else:
            raise ValueError(f"ModelTrainer ({self.model_name}): Modelo '{self.model_name}' no está soportado.")
    
    def _build_pipeline(self, X_sample: pd.DataFrame):
        """Construye el pipeline de preprocesamiento y el modelo."""
        
        exclude_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        X_sample = X_sample.drop(columns=[col for col in exclude_cols if col in X_sample.columns])
        
        numeric_cols = X_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if 'Pclass' in numeric_cols:
            numeric_cols.remove('Pclass')
            categorical_cols.append('Pclass')
            
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop' 
        )

        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])

    def train_and_evaluate(self, X_ref: ray.ObjectRef, y_ref: ray.ObjectRef, test_size: float):
        """
        Obtiene los datos del Object Store, los divide, entrena el modelo y lo evalúa.
        Devuelve el pipeline entrenado y sus metadatos.
        """
        X = ray.get(X_ref) if isinstance(X_ref, ray.ObjectRef) else X_ref
        y = ray.get(y_ref) if isinstance(y_ref, ray.ObjectRef) else y_ref

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logger.info(f"Entrenamiento ({self.model_name}): Datos divididos para entrenamiento/prueba. Train: {X_train.shape}, Test: {X_test.shape}")

        logger.info(f"Entrenamiento ({self.model_name}): Iniciando entrenamiento con parámetros {self.model_params}...")
        
        pipeline = self._build_pipeline(X_train)
        pipeline.fit(X_train, y_train)
        
        logger.info(f"Entrenamiento ({self.model_name}): Modelo entrenado. Evaluando...")

        #Evaluación 
        preds = pipeline.predict(X_test)
        
        try:
            probas = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, probas)
        except AttributeError: 
            probas = None
            roc_auc = None
            logger.warning(f"Evaluación ({self.model_name}): Modelo no soporta predict_proba.")

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1_score": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }
        logger.info(f"Evaluación ({self.model_name}): Completada. Accuracy: {metrics['accuracy']:.4f}")

        metadata = {
            "model_name": self.model_name,
            "training_date": datetime.now().isoformat(),
            "params": self.model_params,
            "metrics": metrics,
        }
        
        return pipeline, metadata

@ray.remote 
def safe_train_and_get_pipeline(name, params, X_ref, y_ref, test_size, retries=3):
    """
    Función remota que encapsula la creación del ModelTrainer y su ejecución,
    con reintentos en caso de fallo del actor.
    Devuelve el pipeline entrenado y sus metadatos.
    """
    for attempt in range(retries):
        try:
            trainer = ModelTrainer.remote(name, params)
            pipeline, metadata = ray.get(trainer.train_and_evaluate.remote(X_ref, y_ref, test_size))
            return name, pipeline, metadata
        except ActorDiedError:
            logger.warning(f"Reintento ({name}): Intento {attempt + 1} fallido por ActorDiedError. Reintentando...")
            time.sleep(2)
        except Exception as e:
            logger.error(f"Reintento ({name}): Intento {attempt + 1} fallido por error inesperado: {e}. Reintentando...")
            time.sleep(2)
            
    logger.error(f"Fallo Total ({name}): El modelo '{name}' falló después de {retries} intentos y no pudo ser entrenado.")
    return name, None, None

def main():
    # Define el namespace que usarán para todos los actores detached
    RAY_NAMESPACE = "my_ml_models_namespace" 

    try:
        context = ray.init(address="ray://ray-head:10001", ignore_reinit_error=True, namespace=RAY_NAMESPACE)
        logger.info(f"¡Conexión a Ray exitosa en el namespace: {RAY_NAMESPACE}!")
    except Exception as e:
        logger.critical(f"Main: ¡Fallo CRÍTICO al inicializar Ray! No se puede continuar. Error: {str(e)}")
        exit(1)

    data_dir_path = "/app/data" 
    
    if not os.path.exists(data_dir_path):
        logger.critical(f"Main: ¡Error crítico! El directorio de datos '{data_dir_path}' no existe en el contenedor. Asegúrate de que tu Dockerfile copia la carpeta 'data/' a /app/data/.")
        exit(1)

    logger.info(f"Main: Iniciando carga y división de datos desde {data_dir_path}...")
    try:
        data_loader = DataLoader.remote(data_dir=data_dir_path)
        X_ref, y_ref, test_size = ray.get(data_loader.get_train_test.remote(target_column='Survived'))
        
        logger.info(f"Main: Datos base cargados y listos en Ray Object Store. X_ref: {X_ref}, y_ref: {y_ref}, test_size: {test_size}")

    except FileNotFoundError as e:
        logger.critical(f"Main: Error crítico al cargar datos: {e}")
        exit(1)
    except ValueError as e:
        logger.critical(f"Main: Error de datos: {e}")
        exit(1)
    except Exception as e:
        logger.critical(f"Main: Error inesperado al cargar o preprocesar datos: {e}", exc_info=True)
        exit(1)
    
    logger.info("Main: Lanzando todos los entrenamientos de modelos en paralelo...")
    futures = []
    for config in MODELS_CONFIG:
        name = config["name"]
        params = config.get("params", {})
        future = safe_train_and_get_pipeline.remote(name, params, X_ref, y_ref, test_size)
        futures.append(future)
                
    logger.info("Main: Esperando a que todos los entrenamientos finalicen y creando ModelServiceActors...")
    
    trained_models_info = []
    for name, pipeline, metadata in ray.get(futures):
        if pipeline:
            logger.info(f"Entrenamiento de '{name}' completado. Creando ModelServiceActor para '{name}'.")
            try:
                # Intenta obtener el actor en el namespace específico, si existe, se usa, si no, se crea.
                model_actor_handle = ray.get_actor(name, namespace=RAY_NAMESPACE)
                logger.info(f"ModelServiceActor '{name}' ya existe en el namespace '{RAY_NAMESPACE}'. Saltando recreación.")
                # Si el actor ya existe y se quiere actualizar llamar a un método de actualización:
                # ray.get(model_actor_handle.update_pipeline_and_metadata.remote(pipeline, metadata))
            except ValueError: # Actor no encontrado en este namespace, crearlo
                model_actor_handle = ModelServiceActor.options(
                    name=name, 
                    lifetime="detached", 
                    max_restarts=-1,
                    namespace=RAY_NAMESPACE # Especifica el namespace aquí también
                ).remote(name, pipeline, metadata)
                logger.info(f"ModelServiceActor '{name}' creado y nombrado en el namespace '{RAY_NAMESPACE}'. Handle: {model_actor_handle}")
                # Forzar una llamada remota para asegurar que el actor está completamente inicializado y registrado en el GCS
                ray.get(model_actor_handle.get_model_name.remote()) 
            
            trained_models_info.append({"name": name, "metadata": metadata})
        else:
            logger.error(f"Fallo al entrenar el modelo '{name}'. No se creará ModelServiceActor.")

    logger.info("\n--- Resumen Final de Modelos Entrenados y Actores Creados ---")
    for info in trained_models_info:
        logger.info(f"Resumen: Modelo '{info['name']}' entrenado y su ModelServiceActor creado. Accuracy: {info['metadata']['metrics']['accuracy']:.4f}")
    
    logger.info("\n--- Verificación Final: Actores de Modelos en el Clúster Ray ---")
    available_actors = []
    for config in MODELS_CONFIG:
        name = config["name"]
        try:
            # Verificar en el namespace correcto
            actor = ray.get_actor(name, namespace=RAY_NAMESPACE)
            meta = ray.get(actor.get_metadata.remote())
            available_actors.append(f"{name} (Accuracy: {meta['metrics']['accuracy']:.4f})")
        except ValueError:
            available_actors.append(f"{name} (NO DISPONIBLE en el namespace '{RAY_NAMESPACE}')")
        except Exception as e:
            available_actors.append(f"{name} (ERROR: {e})")
    
    logger.info("Actores de modelos disponibles en Ray:")
    for actor_info in available_actors:
        logger.info(f"- {actor_info}")
        
if __name__ == "__main__":
    main()
