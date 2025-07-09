import logging
import ray
import uvicorn
from ray_cluster import initialize_ray    
from api_fast import create_app           

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_platform.log')
    ]
)

logger = logging.getLogger(__name__)

def get_model_names():
    try:
        if ray.is_initialized():
            all_actors = ray.util.list_named_actors()
            model_names = [actor for actor in all_actors if not actor.startswith("__")]
            logger.info(f"Found {len(model_names)} model actors: {model_names}")
            return model_names
    except Exception as e:
        logger.warning(f"Could not discover Ray actors: {e}")
    return []

def create_global_app():
    if not ray.is_initialized():
        success = initialize_ray(address="ray-head:6379", local_mode=False)
        if success:
            logger.info("Ray initialized for API serving")
        else:
            logger.warning("Ray could not be initialized. Starting without actors.")
        
    model_names = get_model_names()
    app, predictor = create_app(model_names)
    return app

app = create_global_app()

def main():
    logger.info("Starting API server in standalone mode")
    
    model_names = get_model_names()
    app, predictor = create_app(model_names)
    
    logger.info(f"Starting API server with {len(model_names)} models")
    uvicorn.run(app, host="0.0.0.0", port=8000)
