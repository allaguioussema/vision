import os
from typing import Optional


class Config:
    """Configuration class for the Custom Detection API"""
    
    # Model settings
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
    
    # Performance settings
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    MAX_FILE_SIZE = os.getenv("MAX_FILE_SIZE", "10MB")  # Keep as string for now
    
    # Model paths
    INGREDIENT_MODEL_PATH = os.getenv("INGREDIENT_MODEL_PATH", "models/ingredient.pt")
    NUTRITION_MODEL_PATH = os.getenv("NUTRITION_MODEL_PATH", "models/nutritiontable.pt")
    
    # API settings
    API_TITLE = os.getenv("API_TITLE", "Custom Detection API")
    API_VERSION = os.getenv("API_VERSION", "2.0.0")
    API_DESCRIPTION = os.getenv(
        "API_DESCRIPTION", 
        "Advanced ingredient and nutrition table detection using custom YOLO models"
    )
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Development settings
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    RELOAD = os.getenv("RELOAD", "false").lower() == "true"
    
    @classmethod
    def get_model_paths(cls) -> dict:
        """Get model paths as a dictionary"""
        return {
            "ingredient": cls.INGREDIENT_MODEL_PATH,
            "nutrition": cls.NUTRITION_MODEL_PATH,
        }
    
    @classmethod
    def get_max_file_size_bytes(cls) -> int:
        """Convert MAX_FILE_SIZE string to bytes"""
        size_str = cls.MAX_FILE_SIZE.upper()
        if size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            # Assume bytes if no suffix
            return int(size_str)
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        try:
            # Validate thresholds
            if not (0.0 <= cls.CONFIDENCE_THRESHOLD <= 1.0):
                raise ValueError("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
            
            if not (0.0 <= cls.IOU_THRESHOLD <= 1.0):
                raise ValueError("IOU_THRESHOLD must be between 0.0 and 1.0")
            
            # Validate workers
            if cls.MAX_WORKERS < 1:
                raise ValueError("MAX_WORKERS must be at least 1")
            
            # Validate file size format
            try:
                cls.get_max_file_size_bytes()
            except ValueError:
                raise ValueError("MAX_FILE_SIZE must be in format like '10MB', '1GB', etc.")
            
            # Validate model paths exist
            import os
            for model_type, path in cls.get_model_paths().items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file not found: {path}")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {str(e)}")
            return False


# Global config instance
config = Config() 