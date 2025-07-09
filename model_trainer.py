import ray
import logging
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import random
import socket
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

@ray.remote(max_retries=3)
def train_model(model, X_train, y_train, X_test, y_test, model_name=None):
    """
    Train a single model in a distributed manner using Ray
    
    Args:
        model: Scikit-learn model to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging
        
    Returns:
        dict: Training results including trained model and metrics
    """
    if model_name is None:
        model_name = model.__class__.__name__
    
    # Create a unique ID for this training job
    job_id = f"{model_name}_{socket.gethostname()}_{random.randint(1000, 9999)}"
    
    logger.info(f"Training model: {model_name} (job_id: {job_id})")
    
    start_time = time()
    
    try:
        # Ensure correct input type for KNeighborsClassifier to avoid numpy Flags error
        if isinstance(model, KNeighborsClassifier):
            if hasattr(X_train, 'values'):
                X_train = X_train.values
            if hasattr(X_test, 'values'):
                X_test = X_test.values
            if hasattr(y_train, 'values'):
                y_train = y_train.values
            if hasattr(y_test, 'values'):
                y_test = y_test.values

        # Train the model
        model.fit(X_train, y_train)
        training_time = time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics - Force classification metrics only
        metrics = {}
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
        metrics['training_time'] = training_time
        
        logger.info(f"Training completed for {model_name}: {metrics}")
        
        return {
            'model': model,
            'model_name': model_name,
            'metrics': metrics
        }
    except Exception as e:
        logger.error(f"Error training model {model_name}: {e}")
        return {
            'model': None,
            'model_name': model_name,
            'error': str(e)
        }

def train_multiple_models(models, X_train, y_train, X_test, y_test, model_names=None, max_retries=3, timeout=600):
    """
    Train multiple models in parallel using Ray with fault tolerance
    
    Args:
        models (list): List of scikit-learn models to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_names (list, optional): Names of the models
        
    Returns:
        tuple: (trained_models, accuracy_dict)
    """
    if not ray.is_initialized():
        logger.warning("Ray is not initialized, initializing now")
        ray.init()
    
    if model_names is None:
        model_names = [model.__class__.__name__ for model in models]
    
    logger.info(f"Training {len(models)} models in parallel with fault tolerance")
    
    # Data sharding - store data in Ray's object store for distributed access
    X_train_id = ray.put(X_train)
    y_train_id = ray.put(y_train)
    X_test_id = ray.put(X_test)
    y_test_id = ray.put(y_test)
    
    # Start distributed training with fault tolerance
    training_refs = []
    for i, (model, model_name) in enumerate(zip(models, model_names)):
        # Distribute models across different Ray workers
        model_id = ray.put(model)
        
        # Submit training task with retry logic
        for attempt in range(max_retries):
            try:
                ref = train_model.remote(
                    model_id, X_train_id, y_train_id, X_test_id, y_test_id, 
                    model_name=model_name
                )
                training_refs.append((ref, model_name, attempt))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to submit {model_name} training task (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to submit {model_name} training task after {max_retries} attempts: {e}")
    
    # Collect results with timeout handling
    trained_models = {}
    all_metrics = {}
    
    for ref, model_name, _ in training_refs:
        try:
            # Wait for result with timeout
            result = ray.get(ref, timeout=timeout)
            
            if result.get('error'):
                logger.error(f"Model {model_name} failed: {result['error']}")
                continue
                
            trained_models[model_name] = result['model']
            
            # Store complete metrics for each model
            all_metrics[model_name] = result['metrics']
        except ray.exceptions.GetTimeoutError:
            logger.error(f"Training {model_name} timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Error getting result for {model_name}: {e}")
    
    if not trained_models:
        logger.error("All model training failed")
    else:
        logger.info(f"Successfully trained {len(trained_models)} models")
        
    return trained_models, all_metrics

# Ray actor for distributed, in-memory model storage and prediction
@ray.remote(max_restarts=-1, max_task_retries=3, num_cpus=0.5)
class ModelActor:
    def __init__(self, model=None, model_name=None):
        """
        Initialize the ModelActor with a scikit-learn model and its name.
        
        Args:
            model: The scikit-learn model to be used for predictions (optional).
            model_name: A string representing the name of the model.
        """
        self.model = model
        self.model_name = model_name or "Unknown"
        self.metrics = {}
        self.training_data = {}  # Store training data for plotting
        self.node_id = ray.get_runtime_context().get_node_id()
        logger.info(f"ModelActor {self.model_name} initialized on node {self.node_id}")
    
    def get_health(self):
        """Check if the actor is healthy and return node information"""
        return {
            'model_name': self.model_name,
            'node_id': self.node_id,
            'status': 'healthy'
        }
    
    def predict(self, features):
        """
        Make predictions using the stored model.
        
        Args:
            features: Input features for prediction, as a list of dictionaries or a DataFrame.
            
        Returns:
            List of predictions.
        """
        import pandas as pd
        X = pd.DataFrame(features)
        
        # Ensure columns are in the correct order if the model has feature_names_in_
        if hasattr(self.model, 'feature_names_in_'):
            # Check for missing features and add them with default values (0)
            missing_features = set(self.model.feature_names_in_) - set(X.columns)
            if missing_features:
                logger.warning(f"Missing features in prediction input: {missing_features}. Adding with default value 0.")
                for feature in missing_features:
                    X[feature] = 0
            
            # Check for extra features and remove them
            extra_features = set(X.columns) - set(self.model.feature_names_in_)
            if extra_features:
                logger.warning(f"Extra features in prediction input: {extra_features}. Removing them.")
                X = X.drop(columns=list(extra_features))
            
            # Reorder columns to match training order
            X = X[self.model.feature_names_in_]
        
        preds = self.model.predict(X)
        return preds.tolist()
    
    def get_name(self):
        """
        Get the name of the model.
        
        Returns:
            The model name as a string.
        """
        return self.model_name
    
    def get_metrics(self):
        """
        Get the metrics of the model.
        
        Returns:
            A dictionary containing the model's metrics.
        """
        return self.metrics
    
    def set_metrics(self, metrics):
        """
        Set the metrics for the model.
        
        Args:
            metrics: A dictionary containing the metrics to be set for the model.
        """
        self.metrics = metrics

    def set_training_data(self, X_train, y_train, X_test, y_test):
        """
        Store training data for generating plots.
        
        Args:
            X_train, y_train, X_test, y_test: Training and test datasets
        """
        self.training_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }

    def get_model_info(self):
        """
        Return model info for API listing (/models endpoint).
        
        Returns:
            dict: Model info including name, algorithm, metrics, and features if available.
        """
        info = {
            'name': self.model_name,
            'algorithm': type(self.model).__name__,
            'metrics': self.metrics,
            'features': list(self.model.feature_names_in_) if hasattr(self.model, 'feature_names_in_') else None
        }
        return info

    def generate_confusion_matrix_png(self):
        """
        Generate confusion matrix as PNG image (base64 encoded).
        
        Returns:
            A dictionary containing the PNG image as base64 string.
        """
        try:
            if not self.training_data:
                return {"error": "No training data available for confusion matrix generation"}
            
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            import numpy as np
            import base64
            from io import BytesIO
            
            X_test = self.training_data['X_test']
            y_test = self.training_data['y_test']
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Create the plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=np.unique(y_test), 
                        yticklabels=np.unique(y_test))
            plt.title(f'Matriz de Confusión - {self.model_name}')
            plt.xlabel('Predicción')
            plt.ylabel('Valor Real')
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                'model_name': self.model_name,
                'confusion_matrix_png': image_base64,
                'format': 'png',
                'encoding': 'base64'
            }
        except Exception as e:
            return {'error': f'Failed to generate confusion matrix PNG: {str(e)}'}

    def generate_plot_data(self):
        """
        Generate all plot data for the model (confusion matrix for classification).
        
        Returns:
            A dictionary containing confusion matrix data, plus metrics.
        """
        return {
            'confusion_matrix': self.generate_confusion_matrix_png(),
            'model_name': self.model_name,
            'metrics': self.metrics
        }
    

    def generate_plots_png(self):
        """
        Generate confusion matrix as PNG image for classification models.
        
        Returns:
            A dictionary containing the PNG image as base64 string.
        """
        return {
            'confusion_matrix': self.generate_confusion_matrix_png(),
            'model_name': self.model_name
        }