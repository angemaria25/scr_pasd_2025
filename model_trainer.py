"""
Model training utilities for distributed machine learning with fault tolerance (Ray-native, no file I/O)
"""
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
        
        # Calculate metrics
        metrics = {}
        
        # Classification metrics
        if len(np.unique(y_train)) < 10:  # Assuming it's a classification task if fewer than 10 unique values
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        # Regression metrics
        else:
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
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

    def generate_roc_curve(self):
        """
        Generate ROC curve data for classification models.
        
        Returns:
            Dictionary with ROC curve data or error message
        """
        try:
            if not self.training_data:
                return {"error": "No training data available for ROC curve generation"}
            
            import numpy as np
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            
            X_test = self.training_data['X_test']
            y_test = self.training_data['y_test']
            
            # Check if it's a classification task
            n_classes = len(np.unique(y_test))
            if n_classes < 2:
                return {"error": "ROC curve requires at least 2 classes"}
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_test)
            elif hasattr(self.model, 'decision_function'):
                y_scores = self.model.decision_function(X_test)
                if n_classes == 2:
                    # Binary classification
                    y_proba = np.column_stack([1 - y_scores, y_scores])
                else:
                    return {"error": "Multi-class ROC with decision_function not supported"}
            else:
                return {"error": "Model does not support probability prediction"}
            
            roc_data = {}
            
            if n_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                roc_data = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc,
                    'type': 'binary'
                }
            else:
                # Multi-class classification
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                roc_data = {'type': 'multiclass', 'classes': {}}
                
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_data['classes'][f'class_{i}'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'auc': roc_auc
                    }
            
            return roc_data
            
        except Exception as e:
            return {"error": f"Failed to generate ROC curve: {str(e)}"}

    def generate_learning_curve(self):
        """
        Generate learning curve data.
        
        Returns:
            Dictionary with learning curve data or error message
        """
        try:
            if not self.training_data:
                return {"error": "No training data available for learning curve generation"}
            
            from sklearn.model_selection import learning_curve
            import numpy as np
            
            X_train = self.training_data['X_train']
            y_train = self.training_data['y_train']
            
            # Generate learning curve
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                self.model, X_train, y_train, 
                train_sizes=train_sizes,
                cv=3,  # 3-fold cross-validation
                n_jobs=1,
                random_state=42
            )
            
            # Calculate mean and std
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            val_scores_mean = np.mean(val_scores, axis=1)
            val_scores_std = np.std(val_scores, axis=1)
            
            return {
                'train_sizes': train_sizes_abs.tolist(),
                'train_scores_mean': train_scores_mean.tolist(),
                'train_scores_std': train_scores_std.tolist(),
                'val_scores_mean': val_scores_mean.tolist(),
                'val_scores_std': val_scores_std.tolist()
            }
            
        except Exception as e:
            return {"error": f"Failed to generate learning curve: {str(e)}"}

    def generate_plot_data(self):
        """
        Generate all plot data for the model (ROC curve and learning curve).
        
        Returns:
            A dictionary containing both ROC curve and learning curve data, plus metrics.
        """
        return {
            'roc_curve': self.generate_roc_curve(),
            'learning_curve': self.generate_learning_curve(),
            'model_name': self.model_name,
            'metrics': self.metrics
        }
    
    def generate_roc_png(self):
        """
        Generate ROC curve as PNG image (base64 encoded).
        
        Returns:
            A dictionary containing the PNG image as base64 string.
        """
        try:
            import sys
            import os
            sys.path.append('/app')
            from src.visualization.visualizer import plot_roc_curve_to_png
            
            roc_data = self.generate_roc_curve()
            if 'error' in roc_data:
                return {'error': roc_data['error']}
            
            png_base64 = plot_roc_curve_to_png(roc_data, self.model_name, return_base64=True)
            return {
                'model_name': self.model_name,
                'roc_curve_png': png_base64,
                'format': 'png',
                'encoding': 'base64'
            }
        except Exception as e:
            return {'error': f'Failed to generate ROC curve PNG: {str(e)}'}
    
    def generate_learning_curve_png(self):
        """
        Generate learning curve as PNG image (base64 encoded).
        
        Returns:
            A dictionary containing the PNG image as base64 string.
        """
        try:
            import sys
            import os
            sys.path.append('/app')
            from src.visualization.visualizer import plot_learning_curve_to_png
            
            learning_data = self.generate_learning_curve()
            if 'error' in learning_data:
                return {'error': learning_data['error']}
            
            png_base64 = plot_learning_curve_to_png(learning_data, self.model_name, return_base64=True)
            return {
                'model_name': self.model_name,
                'learning_curve_png': png_base64,
                'format': 'png',
                'encoding': 'base64'
            }
        except Exception as e:
            return {'error': f'Failed to generate learning curve PNG: {str(e)}'}
    
    def generate_plots_png(self):
        """
        Generate both ROC curve and learning curve as PNG images.
        
        Returns:
            A dictionary containing both PNG images as base64 strings.
        """
        return {
            'roc_curve': self.generate_roc_png(),
            'learning_curve': self.generate_learning_curve_png(),
            'model_name': self.model_name
        }
