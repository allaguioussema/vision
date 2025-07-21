from ultralytics import YOLO
from typing import Dict, Optional
import asyncio
import logging
from functools import lru_cache
import numpy as np
from .config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Global model cache
_model_cache: Dict[str, YOLO] = {}


def load_model(model_type: str) -> YOLO:
    """
    Load a specific model with error handling and logging.
    
    Args:
        model_type (str): Type of model to load ("ingredient" or "nutrition")
        
    Returns:
        YOLO: Loaded model instance
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If model loading fails
    """
    model_paths = config.get_model_paths()
    
    if model_type not in model_paths:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_path = model_paths[model_type]
    
    try:
        logger.info(f"Loading {model_type} model from {model_path}")
        model = YOLO(model_path)
        
        # Warm up the model with a dummy image
        logger.info(f"Warming up {model_type} model...")
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        model.predict(
            source=dummy_image, 
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False
        )
        
        logger.info(f"Successfully loaded {model_type} model")
        return model
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load {model_type} model: {str(e)}")
        raise Exception(f"Failed to load {model_type} model: {str(e)}")


def get_model(model_type: str) -> YOLO:
    """
    Get a model from cache or load it if not cached.
    
    Args:
        model_type (str): Type of model to get ("ingredient" or "nutrition")
        
    Returns:
        YOLO: Model instance
    """
    if model_type not in _model_cache:
        _model_cache[model_type] = load_model(model_type)
    return _model_cache[model_type]


def load_all_models() -> Dict[str, YOLO]:
    """
    Load all models and return them in a dictionary.
    
    Returns:
        dict[str, YOLO]: Dictionary of all models
    """
    models = {}
    model_paths = config.get_model_paths()
    
    for model_type in model_paths.keys():
        try:
            models[model_type] = get_model(model_type)
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {str(e)}")
            # Continue loading other models even if one fails
            continue
    
    return models


def get_available_models() -> list[str]:
    """
    Get list of available model types.
    
    Returns:
        list[str]: List of available model types
    """
    return list(config.get_model_paths().keys())


# Pre-load all models at startup for better performance
def initialize_models():
    """
    Initialize all models at startup.
    """
    logger.info("Initializing all models...")
    
    # Validate configuration first
    if not config.validate():
        raise ValueError("Configuration validation failed")
    
    try:
        load_all_models()
        logger.info("All models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise
