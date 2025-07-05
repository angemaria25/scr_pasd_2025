"""
Model serving API for distributed ML platform using FastAPI - Cleaned version
"""
import base64
import io
import json
import logging
import time
import asyncio
from typing import Dict, List, Any

import pandas as pd
import ray
import ray.util
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Pydantic models for request validation
class FileUploadRequest(BaseModel):
    filename: str
    content: str  # Base64 encoded file content
    
class FileUploadResponse(BaseModel):
    filename: str
    status: str
    rows: int = None
    columns: List[str] = None
    preview: List[Dict[str, Any]] = None

def create_app(model_names):
    app = FastAPI(title="Distributed ML Platform API", description="Cleaned API for distributed machine learning")
    
    # Semaphore to limit concurrent upload operations
    upload_semaphore = asyncio.Semaphore(2)  # Allow max 2 concurrent uploads

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/dataset_preview")
    async def dataset_preview(dataset: str, n: int = 5):
        """Return a preview/sample of the dataset by name, loading from persistent storage if needed."""
        from src.data.data_loader import get_data_manager, load_dataset
        import os
        data_manager = get_data_manager()
        preview = data_manager.get_file_sample(dataset, n=n)
        if preview is not None:
            return {"dataset": dataset, "preview": preview}

        # Try to load from persistent storage (search all likely locations)
        search_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "datasets"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..", "datasets")
        ]
        found_path = None
        for d in search_dirs:
            candidate = os.path.join(d, dataset)
            if os.path.exists(candidate):
                found_path = candidate
                break
        if found_path:
            df = load_dataset(found_path)
            if df is not None:
                data_manager.store_file_data(dataset, df.to_dict('records'))
                preview = data_manager.get_file_sample(dataset, n=n)
                if preview is not None:
                    return {"dataset": dataset, "preview": preview}
                else:
                    return {"dataset": dataset, "preview": [], "error": "Failed to get preview after loading."}
            else:
                return {"dataset": dataset, "preview": [], "error": "Failed to load dataset from disk."}
        return {"dataset": dataset, "preview": [], "error": "Dataset not found in object store or persistent storage."}

    @app.get("/")
    async def root():
        return {"message": "Distributed ML Platform API", "models_available": len(model_names)}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "models_loaded": len(model_names)}

    @app.post("/clear_memory")
    async def clear_distributed_memory():
        """Clear all distributed memory including trained models and datasets"""
        try:
            logger.info("Starting distributed memory cleanup...")
            
            # Clear all named actors (trained models)
            all_actors = ray.util.list_named_actors()
            cleared_actors = 0
            
            for actor_name in all_actors:
                try:
                    actor = ray.get_actor(actor_name)
                    ray.kill(actor)
                    cleared_actors += 1
                    logger.info(f"Cleared actor: {actor_name}")
                except Exception as e:
                    logger.warning(f"Could not clear actor {actor_name}: {e}")
            
            # Clear object store data
            from src.data.data_loader import get_data_manager
            data_manager = get_data_manager()
            
            try:
                # Clear all stored datasets
                files_cleared = data_manager.clear_all_data()
                logger.info(f"Cleared {files_cleared} datasets from object store")
            except Exception as e:
                logger.warning(f"Could not clear object store: {e}")
                files_cleared = 0
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info(f"Memory cleanup completed. Cleared {cleared_actors} actors and {cleared_actors} datasets")
            
            return {
                "status": "success",
                "actors_cleared": cleared_actors,
                "datasets_cleared": files_cleared,
                "message": "Distributed memory cleared successfully"
            }
            
        except Exception as e:
            logger.error(f"Error clearing distributed memory: {e}")
            raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

    # Models endpoints
    @app.get("/models")
    async def models():
        """List all available models"""
        try:
            # Get list of all named actors that are models
            all_actors = ray.util.list_named_actors()
            model_actors = []
            
            for actor_name in all_actors:
                try:
                    # Try to get the actor and check if it has model methods
                    actor = ray.get_actor(actor_name)
                    # Check if this actor has the methods we expect from a model
                    model_info = ray.get(actor.get_model_info.remote(), timeout=2.0)
                    model_actors.append(model_info)
                except Exception:
                    # Skip actors that don't have model methods or are not accessible
                    continue
            
            return model_actors
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    @app.get("/models/{model_name}/metrics")
    async def model_metrics(model_name: str):
        """Get metrics for a specific model"""
        try:
            actor = ray.get_actor(model_name)
            metrics = ray.get(actor.get_metrics.remote())
            return {"model_name": model_name, "metrics": metrics}
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")

    # Visualization endpoints  
    @app.get("/visualization/{model_name}/roc_curve")
    async def get_roc_curve(model_name: str):
        """Get ROC curve as a viewable PNG image"""
        try:
            actor = ray.get_actor(model_name)
            # Check if model is classification by inspecting metrics or actor info
            try:
                metrics = ray.get(actor.get_metrics.remote(), timeout=2.0)
                # If accuracy is not present, it's likely regression
                if 'accuracy' not in metrics:
                    raise HTTPException(status_code=404, detail=f"ROC curve not available for regression models.")
            except Exception:
                # If metrics can't be fetched, fallback to trying ROC
                pass
            roc_png_data = ray.get(actor.generate_roc_png.remote())
            if 'error' in roc_png_data:
                raise HTTPException(status_code=404, detail=roc_png_data['error'])
            png_bytes = base64.b64decode(roc_png_data['roc_curve_png'])
            return StreamingResponse(
                io.BytesIO(png_bytes),
                media_type="image/png",
                headers={"Content-Disposition": f"inline; filename={model_name}_roc_curve.png"}
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found or ROC not available: {str(e)}")

    @app.get("/visualization/{model_name}/learning_curve")
    async def get_learning_curve(model_name: str):
        """Get learning curve as a viewable PNG image"""
        try:
            actor = ray.get_actor(model_name)
            learning_png_data = ray.get(actor.generate_learning_curve_png.remote())
            
            if 'error' in learning_png_data:
                raise HTTPException(status_code=500, detail=learning_png_data['error'])
            
            # Decode base64 to binary PNG data
            png_bytes = base64.b64decode(learning_png_data['learning_curve_png'])
            
            return StreamingResponse(
                io.BytesIO(png_bytes),
                media_type="image/png",
                headers={"Content-Disposition": f"inline; filename={model_name}_learning_curve.png"}
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found: {str(e)}")

    # Prediction endpoints
    @app.post("/predict")
    async def predict_single(request: Dict[str, Any]):
        """Make a single prediction"""
        try:
            model_name = request.get('model_name')
            features = request.get('features')
            
            if not model_name or not features:
                raise HTTPException(status_code=400, detail="model_name and features are required")
            
            actor = ray.get_actor(model_name)
            prediction = ray.get(actor.predict.remote([features]))
            
            return {
                "model": model_name,
                "prediction": prediction[0] if isinstance(prediction, list) else prediction
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict_batch")
    async def predict_batch(request: Dict[str, Any]):
        """Make batch predictions"""
        try:
            model_name = request.get('model_name')
            data = request.get('data')
            
            if not model_name or not data:
                raise HTTPException(status_code=400, detail="model_name and data are required")
            
            actor = ray.get_actor(model_name)
            predictions = ray.get(actor.predict.remote(data))
            
            return {
                "model": model_name,
                "predictions": predictions
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # File upload endpoint
    @app.post("/upload", response_model=FileUploadResponse)
    async def upload_file(request: FileUploadRequest):
        """Upload and process a file in a distributed manner"""
        async with upload_semaphore:
            try:
                logger.info(f"Upload request received for file: {request.filename}")
                
                # Validate request
                if not request.filename or not request.content:
                    raise HTTPException(status_code=400, detail="Filename and content are required")
                
                # Decode base64 content
                try:
                    content_bytes = base64.b64decode(request.content)
                    logger.info(f"Successfully decoded base64 content for {request.filename} ({len(content_bytes)} bytes)")
                except Exception as e:
                    logger.error(f"Failed to decode base64 content: {e}")
                    raise HTTPException(status_code=400, detail=f"Invalid base64 content: {str(e)}")
                
                # Determine file type and read accordingly
                filename_lower = request.filename.lower()
                if filename_lower.endswith('.csv'):
                    try:
                        df = pd.read_csv(io.StringIO(content_bytes.decode('utf-8')))
                        logger.info(f"Successfully parsed CSV file with {len(df)} rows and {len(df.columns)} columns")
                    except Exception as e:
                        logger.error(f"Failed to parse CSV: {e}")
                        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
                elif filename_lower.endswith('.json'):
                    try:
                        df = pd.read_json(io.StringIO(content_bytes.decode('utf-8')))
                        logger.info(f"Successfully parsed JSON file with {len(df)} rows and {len(df.columns)} columns")
                    except Exception as e:
                        logger.error(f"Failed to parse JSON: {e}")
                        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {request.filename}. Only CSV and JSON files are supported.")
                
                # Store data using object store
                from src.data.data_loader import get_data_manager
                data_manager = get_data_manager()
                
                try:
                    success = data_manager.store_file_data(request.filename, df.to_dict('records'))
                    if not success:
                        raise HTTPException(status_code=500, detail="Failed to store data in object store")
                    logger.info(f"Data stored in object store: {len(df)} records for {request.filename}")
                except Exception as e:
                    logger.error(f"Failed to store data in object store: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to distribute data: {str(e)}")
                
                # Generate preview
                preview = df.head(5).to_dict('records')
                
                response = FileUploadResponse(
                    filename=request.filename,
                    status="uploaded_and_distributed",
                    rows=len(df),
                    columns=list(df.columns),
                    preview=preview
                )
                
                logger.info(f"Successfully processed file {request.filename}")
                return response
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Unexpected error uploading file {request.filename}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    @app.get("/uploaded_files")
    async def list_uploaded_files():
        """List all uploaded files stored in Ray object store"""
        try:
            from src.data.data_loader import get_data_manager
            data_manager = get_data_manager()
            
            uploaded_files_list = data_manager.list_files()
            uploaded_files = []
            
            logger.info(f"Found {len(uploaded_files_list)} files in object store: {uploaded_files_list}")
            
            for filename in uploaded_files_list:
                try:
                    file_info = data_manager.get_file_info(filename)
                    if file_info:
                        uploaded_files.append({
                            'filename': filename,
                            'rows': file_info.get('rows', 0),
                            'columns': file_info.get('columns', []),
                            'loaded_at': file_info.get('loaded_at', 'Unknown')
                        })
                        logger.info(f"Successfully retrieved info for {filename}")
                    else:
                        logger.warning(f"No info found for {filename}")
                except Exception as e:
                    logger.error(f"Error getting info for {filename}: {e}")
                    uploaded_files.append({
                        'filename': filename,
                        'error': str(e)
                    })
            
            logger.info(f"Returning {len(uploaded_files)} uploaded files")
            return {'files': [f['filename'] for f in uploaded_files], 'uploaded_files': uploaded_files}
        except Exception as e:
            logger.error(f"Error listing uploaded files: {e}")
            return {'error': str(e), 'uploaded_files': []}

    # Cluster status endpoints
    @app.get("/cluster/status")
    async def cluster_status():
        """Get Ray cluster status"""
        try:
            if not ray.is_initialized():
                return {
                    "error": "Ray has not been started yet.",
                    "status": "not_initialized"
                }
            
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            nodes = ray.nodes()
            
            active_nodes = len([node for node in nodes if node['Alive']])
            total_cpus_raw = cluster_resources.get('CPU', 0)
            available_cpus_raw = available_resources.get('CPU', 0)
            
            # Cap CPU values to be more realistic
            total_cpus_realistic = 0
            available_cpus_realistic = 0
            
            for i, node in enumerate(nodes):
                if node.get('Alive'):
                    node_cpu_raw = node.get('Resources', {}).get('CPU', 2.0)
                    node_cpu_cap = min(node_cpu_raw, 8 if i == 0 else 4)
                    total_cpus_realistic += node_cpu_cap
                    
                    if total_cpus_raw > 0:
                        proportion_available = available_cpus_raw / total_cpus_raw
                        available_cpus_realistic += node_cpu_cap * proportion_available
            
            total_memory = cluster_resources.get('memory', 0)
            available_memory = available_resources.get('memory', 0)
            
            return {
                "status": "healthy",
                "cluster_resources": {
                    **cluster_resources,
                    "CPU_realistic": total_cpus_realistic
                },
                "available_resources": {
                    **available_resources,
                    "CPU_realistic": available_cpus_realistic
                },
                "nodes": len(nodes),
                "active_nodes": active_nodes,
                "node_details": nodes,
                "summary": {
                    "total_cpus": total_cpus_realistic,
                    "available_cpus": available_cpus_realistic,
                    "cpu_utilization": round((total_cpus_realistic - available_cpus_realistic) / total_cpus_realistic * 100, 2) if total_cpus_realistic > 0 else 0,
                    "total_memory_gb": round(total_memory / 1e9, 2),
                    "available_memory_gb": round(available_memory / 1e9, 2),
                    "memory_utilization": round((total_memory - available_memory) / total_memory * 100, 2) if total_memory > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Error getting cluster status: {e}")
            return {
                "error": str(e), 
                "status": "error"
            }

    @app.get("/cluster/workers")
    async def get_workers():
        """Get worker information from Ray cluster"""
        try:
            # Get actual Ray cluster state
            nodes = ray.nodes()
            alive_nodes = [n for n in nodes if n.get("Alive", False)]
            
            # Head node is typically the first one or has specific characteristics
            head_node = alive_nodes[0] if alive_nodes else None
            worker_nodes = alive_nodes[1:] if len(alive_nodes) > 1 else []
            
            worker_details = []
            for i, worker_node in enumerate(worker_nodes):
                node_resources = worker_node.get("Resources", {})
                worker_details.append({
                    "number": i + 1,
                    "name": f"ray_worker_{i + 1}",
                    "status": "running",
                    "node_id": worker_node.get("NodeID", "unknown"),
                    "resources": {
                        "CPU": node_resources.get("CPU", 0),
                        "memory": node_resources.get("memory", 0)
                    }
                })
            
            return {
                "success": True,
                "workers": worker_details,
                "total_workers": len(worker_details),
                "total_nodes": len(alive_nodes)
            }
        except Exception as e:
            logger.error(f"Error getting worker details: {e}")
            return {"error": str(e), "success": False}

    # Training endpoint
    @app.post("/train_all_datasets")
    async def train_all_datasets_endpoint(request: Dict[str, Any]):
        """Train multiple models across multiple datasets in a batch"""
        try:
            datasets_config = request.get('datasets', {})
            
            if not datasets_config:
                raise HTTPException(status_code=400, detail="No datasets configuration provided")
            
            # Import training function
            from src.models.model_trainer import train_multiple_models, ModelActor
            from src.data.data_loader import get_data_manager
            
            batch_results = {}
            total_models_trained = 0
            
            for filename, config in datasets_config.items():
                try:
                    # Get configuration for this dataset
                    task_type = config.get('task_type')
                    target_column = config.get('target_column')
                    algorithms = config.get('algorithms', [])
                    test_size = config.get('test_size', 0.2)
                    random_state = config.get('random_state', 42)
                    
                    if not task_type or not target_column or not algorithms:
                        batch_results[filename] = {
                            "status": "error",
                            "error": "Missing required fields"
                        }
                        continue
                    
                    # Get data from Ray object store
                    try:
                        data_manager = get_data_manager()
                        dataset_records = data_manager.get_file_data(filename)
                        
                        if not dataset_records:
                            batch_results[filename] = {
                                "status": "error",
                                "error": f"No data found for file: {filename}"
                            }
                            continue
                        
                        df = pd.DataFrame(dataset_records)
                        
                    except Exception as e:
                        logger.error(f"Error loading data for {filename}: {e}")
                        batch_results[filename] = {
                            "status": "error", 
                            "error": f"Data loading failed for {filename}: {str(e)}"
                        }
                        continue
                    
                    # Convert task name to classification/regression
                    is_classification = task_type == "classification"
                    
                    # Split features and target
                    if target_column not in df.columns:
                        batch_results[filename] = {
                            "status": "error",
                            "error": f"Target column not found: {target_column}"
                        }
                        continue
                    
                    X = df.drop(target_column, axis=1)
                    y = df[target_column]
                    
                    # Auto-detect and handle target variable type for classification
                    if is_classification:
                        # Check if target is continuous and needs to be converted
                        unique_values = y.nunique()
                        is_numeric = pd.api.types.is_numeric_dtype(y)
                        
                        logger.info(f"Target column '{target_column}' analysis: {unique_values} unique values, numeric: {is_numeric}")
                        
                        if is_numeric and unique_values > 10:
                            # Likely continuous variable - convert to binary classification or suggest regression
                            logger.warning(f"Target '{target_column}' appears to be continuous ({unique_values} unique values). Converting to binary classification.")
                            median_value = y.median()
                            y = (y > median_value).astype(int)
                            logger.info(f"Converted target to binary: 0 (≤{median_value}), 1 (>{median_value})")
                        elif is_numeric:
                            # Make sure it's integer type for classification
                            y = y.astype(int)
                    
                    # Convert categorical features to numeric
                    X = pd.get_dummies(X)
                    
                    # Train split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                    
                    # Define available models based on task type
                    if is_classification:
                        from sklearn.ensemble import GradientBoostingClassifier
                        from sklearn.linear_model import LogisticRegression
                        from sklearn.svm import SVC
                        from sklearn.neighbors import KNeighborsClassifier
                        from sklearn.tree import DecisionTreeClassifier
                        
                        model_mapping = {
                            "decision_tree": DecisionTreeClassifier(random_state=random_state),
                            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
                            "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
                            "svm": SVC(probability=True, random_state=random_state),
                            "k_nearest_neighbors": KNeighborsClassifier(n_neighbors=5)
                        }
                    else:
                        from sklearn.ensemble import GradientBoostingRegressor
                        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
                        from sklearn.tree import DecisionTreeRegressor
                        
                        model_mapping = {
                            "decision_tree_regressor": DecisionTreeRegressor(random_state=random_state),
                            "gradient_boosting_regressor": GradientBoostingRegressor(random_state=random_state),
                            "linear_regression": LinearRegression(),
                            "ridge_regression": Ridge(random_state=random_state),
                            "lasso_regression": Lasso(random_state=random_state),
                            "elastic_net": ElasticNet(random_state=random_state)
                        }
                    
                    # Filter models based on user selection
                    dataset_name = filename.replace('.csv', '').replace('.json', '')
                    models = []
                    model_names_list = []
                    
                    for algo in algorithms:
                        if algo in model_mapping:
                            models.append(model_mapping[algo])
                            model_names_list.append(f"{algo}_{dataset_name}")
                    
                    if not models:
                        batch_results[filename] = {
                            "status": "error",
                            "error": "No valid algorithms selected"
                        }
                        continue
                    
                    # Start distributed training for this dataset
                    trained_models, training_metrics = train_multiple_models(
                        models, X_train, y_train, X_test, y_test, model_names_list
                    )
                    
                    # Create Ray actors for this dataset
                    dataset_results = {}
                    for model_name, model in trained_models.items():
                        try:
                            actor = ModelActor.options(name=model_name, lifetime="detached").remote(model, model_name)
                            actor.set_metrics.remote(training_metrics.get(model_name, {}))
                            actor.set_training_data.remote(X_train, y_train, X_test, y_test)
                            
                            # Prepare result for this model
                            metrics = training_metrics.get(model_name, {})
                            dataset_results[model_name] = {
                                "model_id": model_name,
                                "accuracy": metrics.get('accuracy', 'N/A'),
                                "metrics": metrics
                            }
                            total_models_trained += 1
                            
                        except Exception as e:
                            dataset_results[model_name] = {
                                "error": str(e),
                                "model_id": model_name
                            }
                    
                    batch_results[filename] = {
                        "status": "success",
                        "models_trained": len(trained_models),
                        "results": dataset_results
                    }
                    
                except Exception as e:
                    batch_results[filename] = {
                        "status": "error",
                        "error": str(e)
                    }
            
            return {
                "status": "batch_training_completed",
                "total_models_trained": total_models_trained,
                "datasets_processed": len(datasets_config),
                "results": batch_results
            }
            
        except Exception as e:
            logger.error(f"Batch training error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app, None


