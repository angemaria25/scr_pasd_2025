"""
Ray utility functions for distributed computing
"""
import ray
import logging
import time

logger = logging.getLogger(__name__)

def initialize_ray(address=None, local_mode=False):
    """
    Initialize Ray with the given configuration
    
    Args:
        address: Optional address to connect to an existing Ray cluster
        local_mode: Whether to initialize Ray in local mode
    
    Returns:
        bool: True if Ray was initialized successfully
    """
    try:
        if ray.is_initialized():
            logger.info("Ray is already initialized")
            return True
            
        runtime_env = {
            "env_vars": {
                "RAY_ENABLE_AUTO_RECONNECT": "1",
            }
        }
        
        namespace = "distributed-ml"
        
        if address:
            logger.info(f"Connecting to Ray cluster at {address}")
            ray.init(address=address, runtime_env=runtime_env, namespace=namespace)
        elif local_mode:
            logger.info("Initializing Ray in local mode")
            ray.init(runtime_env=runtime_env, namespace=namespace)
        else:
            logger.info("Auto-discovering Ray cluster")
            ray.init(address="auto", runtime_env=runtime_env, namespace=namespace)
                
        logger.info(f"Ray initialized with resources: {ray.cluster_resources()}")
        return True
    except Exception as e:
        logger.error(f"Error initializing Ray: {e}")
        return False

def shutdown_ray():
    """Shutdown Ray"""
    try:
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray has been shutdown")
        return True
    except Exception as e:
        logger.error(f"Error shutting down Ray: {e}")
        return False

def get_cluster_status():
    """Get status information about the Ray cluster"""
    try:
        if not ray.is_initialized():
            return None
            
        nodes = ray.nodes()
        total_cpus = 0
        total_memory = 0
        alive_nodes = 0
        
        for node in nodes:
            if node.get('alive', False):
                alive_nodes += 1
                resources = node.get('Resources', {})
                total_cpus += resources.get('CPU', 0)
                total_memory += resources.get('memory', 0) / (1024 * 1024 * 1024)  # GB
                
        return {
            'total_nodes': len(nodes),
            'alive_nodes': alive_nodes,
            'total_cpus': total_cpus,
            'total_memory_gb': round(total_memory, 2),
        }
    except Exception as e:
        logger.error(f"Error getting cluster status: {e}")
        return None

def create_actor(actor_cls, *args, **kwargs):
    """Create a Ray actor with basic fault tolerance"""
    return ray.remote(
        max_retries=3,
        num_cpus=kwargs.pop("num_cpus", 0.1),
    )(actor_cls).remote(*args, **kwargs)
