from ultralytics import YOLO
from typing import Dict


# Prefix for YOLO model file paths
YOLO_MODEL_PREFIX = "../models/yolo11"

MODEL_PATHS = {
    "n": f"{YOLO_MODEL_PREFIX}n.pt",
    "s": f"{YOLO_MODEL_PREFIX}s.pt",
    "m": f"{YOLO_MODEL_PREFIX}m.pt",
}


def load_models() -> Dict[str, YOLO]:
    """
    Load YOLO models from disk and return them in a dictionary.

    Returns:
        dict[str, YOLO]: Dictionary of models keyed by type ("n", "s", "m")
    """
    models = {}
    for model_type, path in MODEL_PATHS.items():
        models[model_type] = YOLO(path)
    return models
