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
from typing import List

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

# Define el namespace que usarán para todos los actores detached
RAY_NAMESPACE = "my_ml_models_namespace"

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

    def update_pipeline_and_metadata(self, new_pipeline: Pipeline, new_metadata: dict):
        """
        Actualiza el pipeline y los metadatos del modelo.
        Útil para reentrenamientos sin reiniciar el actor.
        """
        self.pipeline = new_pipeline
        self.metadata = new_metadata
        logger.info(f"ModelServiceActor '{self.model_name}': Pipeline y metadatos actualizados tras reentrenamiento.")

# ¡DataLoader ha sido eliminado ya que los datos se pasarán directamente a través de ObjectRef!

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
        X_sample = X_sample.drop(columns=[col for col in exclude_cols if col in X_sample.columns], errors='ignore')

        numeric_cols = X_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()

        # Manejar 'Pclass' como categórica si se desea
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
        X = ray.get(X_ref) # Obtener DataFrame directamente del almacén de objetos
        y = ray.get(y_ref) # Obtener Serie directamente del almacén de objetos

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

@ray.remote(max_restarts=-1, max_task_retries=-1)
class TrainingOrchestrator:
    def __init__(self):
        logger.info("TrainingOrchestrator: Inicializado.")
        self.training_tasks = {} # {task_id: {"status": ..., "message": ..., "future": ...}}

    def start_training(self, data_ref: ray.ObjectRef, target_column: str, models_to_train: List[str]):
        """
        Inicia un nuevo proceso de entrenamiento con los datos (referencia a Object Store)
        y la columna objetivo especificados.
        """
        task_id = f"training_{datetime.now().strftime('%Y%m%d%H%M%S_%f')}"
        logger.info(f"TrainingOrchestrator ({task_id}): Iniciando entrenamiento con dataset ObjectRef, target: {target_column}, models: {models_to_train}")

        all_models_config_dict = {config["name"]: config for config in MODELS_CONFIG}
        selected_models_config = [all_models_config_dict[name] for name in models_to_train if name in all_models_config_dict]

        if not selected_models_config:
            logger.error(f"TrainingOrchestrator ({task_id}): Ningún modelo seleccionado para entrenar es válido.")
            self.training_tasks[task_id] = {"status": "error", "message": "Ningún modelo válido seleccionado para entrenar."}
            return {"task_id": task_id, "status": "error", "message": "Ningún modelo válido seleccionado para entrenar."}

        try:
            # Obtener el DataFrame del almacén de objetos
            data_df = ray.get(data_ref)

            if target_column not in data_df.columns:
                raise ValueError(f"El dataset debe contener la columna objetivo '{target_column}'. Columnas disponibles: {data_df.columns.tolist()}")

            X = data_df.drop(target_column, axis=1)
            y = data_df[target_column]

            # Poner X y y de nuevo en el almacén de objetos para pasar ObjectRefs a los trainers
            X_ref = ray.put(X)
            y_ref = ray.put(y)
            test_size = 0.2 # Esto podría ser configurable si se desea

        except Exception as e:
            logger.error(f"TrainingOrchestrator ({task_id}): Error al procesar los datos de ObjectRef: {e}")
            self.training_tasks[task_id] = {"status": "error", "message": f"Error al cargar/procesar los datos: {str(e)}"}
            return {"task_id": task_id, "status": "error", "message": f"Error al cargar/procesar los datos: {str(e)}"}


        futures = []
        for config in selected_models_config:
            name = config["name"]
            params = config.get("params", {})
            future = safe_train_and_get_pipeline.remote(name, params, X_ref, y_ref, test_size)
            futures.append(future)

        processing_future = ray.remote(self._process_training_results).remote(futures)
        self.training_tasks[task_id] = {
            "status": "in_progress",
            "message": "Entrenamiento de modelos iniciado en segundo plano.",
            "future": processing_future,
            "models": models_to_train,
            "target_column": target_column,
            "data_ref": data_ref # Mantener referencia a los datos para seguimiento, aunque no se use directamente aquí
        }

        return {"task_id": task_id, "status": "initiated", "message": "Entrenamiento de modelos iniciado."}

    async def _process_training_results(self, futures: List[ray.ObjectRef]):
        """
        Procesa los resultados de los entrenamientos y registra los actores de modelos.
        Esta es una tarea asíncrona que correrá en segundo plano.
        """
        trained_models_info = []
        for name, pipeline, metadata in ray.get(futures):
            if pipeline:
                logger.info(f"Entrenamiento de '{name}' completado. Creando/Actualizando ModelServiceActor para '{name}'.")
                try:
                    model_actor_handle = ray.get_actor(name, namespace=RAY_NAMESPACE)
                    ray.get(model_actor_handle.update_pipeline_and_metadata.remote(pipeline, metadata))
                    logger.info(f"ModelServiceActor '{name}' actualizado en el namespace '{RAY_NAMESPACE}'.")
                except ValueError: # Actor no encontrado en este namespace, crearlo
                    model_actor_handle = ModelServiceActor.options(
                        name=name,
                        lifetime="detached",
                        max_restarts=-1,
                        namespace=RAY_NAMESPACE
                    ).remote(name, pipeline, metadata)
                    logger.info(f"ModelServiceActor '{name}' creado y nombrado en el namespace '{RAY_NAMESPACE}'. Handle: {model_actor_handle}")
                    ray.get(model_actor_handle.get_model_name.remote()) # Forzar una llamada remota para asegurar que el actor está completamente inicializado

                trained_models_info.append({"name": name, "metadata": metadata})
            else:
                logger.error(f"Fallo al entrenar el modelo '{name}'. No se creará/actualizará ModelServiceActor.")

        logger.info("\n--- Resumen Final de Modelos Entrenados y Actores Creados ---")
        for info in trained_models_info:
            logger.info(f"Resumen: Modelo '{info['name']}' entrenado y su ModelServiceActor creado/actualizado. Accuracy: {info['metadata']['metrics']['accuracy']:.4f}")

        # Ya no hay que eliminar archivos temporales
        return {"status": "completed", "message": "Todos los entrenamientos finalizados.", "trained_models": trained_models_info}

    def get_training_status(self, task_id: str):
        """Devuelve el estado de una tarea de entrenamiento específica."""
        task_info = self.training_tasks.get(task_id)
        if not task_info:
            return {"status": "not_found", "message": f"Tarea de entrenamiento con ID '{task_id}' no encontrada."}

        if task_info["status"] == "in_progress":
            if task_info["future"].is_finished():
                try:
                    result = ray.get(task_info["future"])
                    task_info["status"] = "completed"
                    task_info["message"] = result.get("message", "Entrenamiento completado.")
                    task_info["result"] = result
                except Exception as e:
                    task_info["status"] = "failed"
                    task_info["message"] = f"El entrenamiento falló: {str(e)}"
                    logger.error(f"TrainingOrchestrator: Tarea {task_id} falló: {e}")
            else:
                task_info["message"] = "Entrenamiento en curso..."
        return task_info

def main():
    try:
        context = ray.init(address="ray://ray-head:10001", ignore_reinit_error=True, namespace=RAY_NAMESPACE)
        logger.info(f"¡Conexión a Ray exitosa en el namespace: {RAY_NAMESPACE}!")
    except Exception as e:
        logger.critical(f"Main: ¡Fallo CRÍTICO al inicializar Ray! No se puede continuar. Error: {str(e)}")
        exit(1)

    try:
        orchestrator = ray.get_actor("training_orchestrator", namespace=RAY_NAMESPACE)
        logger.info("TrainingOrchestrator ya existe. Reutilizando.")
    except ValueError:
        orchestrator = TrainingOrchestrator.options(
            name="training_orchestrator",
            lifetime="detached",
            max_restarts=-1,
            namespace=RAY_NAMESPACE
        ).remote()
        logger.info("TrainingOrchestrator creado y registrado como detached.")
        ray.get(orchestrator.get_training_status.remote("non_existent_task_id"))

    logger.info("train.py ha inicializado el clúster Ray y los actores principales. El contenedor permanecerá activo.")

if __name__ == "__main__":
    main()