from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import io
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import json

from .model_loader import get_model, get_available_models, initialize_models
from .image_processor import process_image, get_detection_results, DetectionResult
from .stream_processor import generate_stream, save_temp_video, get_video_path
from .config import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Create FastAPI app with optimized settings
app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for better API accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 template engine setup
templates = Jinja2Templates(directory="templates")

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)

# Pydantic models for API requests/responses
class DetectionRequest(BaseModel):
    model_type: str
    confidence_threshold: Optional[float] = config.CONFIDENCE_THRESHOLD
    iou_threshold: Optional[float] = config.IOU_THRESHOLD

class DetectionResponse(BaseModel):
    model_type: str
    processing_time: float
    detections_count: int
    bboxes: List[List[float]]
    labels: List[str]
    confidences: List[float]
    success: bool
    error_message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    uptime: float
    version: str

# Global variables
startup_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize models and resources on startup"""
    logger.info(f"Starting {config.API_TITLE} v{config.API_VERSION}...")
    try:
        # Initialize all models
        initialize_models()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}")
        raise


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main upload form page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        available_models = get_available_models()
        uptime = time.time() - startup_time
        
        return HealthResponse(
            status="healthy",
            available_models=available_models,
            uptime=uptime,
            version=config.API_VERSION
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.get("/models", response_model=List[str])
async def get_models():
    """Get list of available models"""
    return get_available_models()


@app.post("/detect/ingredient")
async def detect_ingredients(
    file: UploadFile = File(...),
    confidence: float = Query(config.CONFIDENCE_THRESHOLD, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(config.IOU_THRESHOLD, ge=0.0, le=1.0, description="IoU threshold")
):
    """
    Detect ingredients in an uploaded image.
    
    Args:
        file: Image file (JPG, PNG, JPEG)
        confidence: Confidence threshold for detections
        iou: IoU threshold for non-maximum suppression
        
    Returns:
        Annotated image with ingredient detections
    """
    return await _detect_with_model(file, "ingredient", confidence, iou)


@app.post("/detect/nutrition")
async def detect_nutrition_tables(
    file: UploadFile = File(...),
    confidence: float = Query(config.CONFIDENCE_THRESHOLD, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(config.IOU_THRESHOLD, ge=0.0, le=1.0, description="IoU threshold")
):
    """
    Detect nutrition tables in an uploaded image.
    
    Args:
        file: Image file (JPG, PNG, JPEG)
        confidence: Confidence threshold for detections
        iou: IoU threshold for non-maximum suppression
        
    Returns:
        Annotated image with nutrition table detections
    """
    return await _detect_with_model(file, "nutrition", confidence, iou)


@app.post("/detect/both")
async def detect_both(
    file: UploadFile = File(...),
    confidence: float = Query(config.CONFIDENCE_THRESHOLD, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(config.IOU_THRESHOLD, ge=0.0, le=1.0, description="IoU threshold")
):
    """
    Detect both ingredients and nutrition tables in an uploaded image.
    
    Args:
        file: Image file (JPG, PNG, JPEG)
        confidence: Confidence threshold for detections
        iou: IoU threshold for non-maximum suppression
        
    Returns:
        JSON response with detection results from both models
    """
    return await _detect_both_models(file, confidence, iou)


@app.post("/detect/json")
async def detect_json(
    file: UploadFile = File(...),
    model_type: str = Query(..., description="Model type: 'ingredient' or 'nutrition'"),
    confidence: float = Query(config.CONFIDENCE_THRESHOLD, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(config.IOU_THRESHOLD, ge=0.0, le=1.0, description="IoU threshold")
):
    """
    Get detection results as JSON without image annotations.
    
    Args:
        file: Image file (JPG, PNG, JPEG)
        model_type: Type of model to use
        confidence: Confidence threshold for detections
        iou: IoU threshold for non-maximum suppression
        
    Returns:
        JSON response with detection data
    """
    return await _detect_json(file, model_type, confidence, iou)


async def _detect_with_model(
    file: UploadFile, 
    model_type: str, 
    confidence: float, 
    iou: float
) -> StreamingResponse:
    """Internal function to handle detection with a specific model"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload JPG, JPEG, or PNG images."
        )
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Get model
        model = get_model(model_type)
        
        # Run detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            process_image, 
            file_bytes, 
            model, 
            model_type
        )
        
        return StreamingResponse(
            io.BytesIO(result), 
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=detected_{model_type}.jpg"}
        )
        
    except Exception as e:
        logger.error(f"Error in {model_type} detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


async def _detect_both_models(
    file: UploadFile, 
    confidence: float, 
    iou: float
) -> JSONResponse:
    """Internal function to handle detection with both models"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload JPG, JPEG, or PNG images."
        )
    
    try:
        # Read file bytes once
        file_bytes = await file.read()
        
        # Get both models
        ingredient_model = get_model("ingredient")
        nutrition_model = get_model("nutrition")
        
        # Run both detections concurrently
        loop = asyncio.get_event_loop()
        
        ingredient_task = loop.run_in_executor(
            executor, 
            get_detection_results, 
            file_bytes, 
            ingredient_model, 
            "ingredient"
        )
        
        nutrition_task = loop.run_in_executor(
            executor, 
            get_detection_results, 
            file_bytes, 
            nutrition_model, 
            "nutrition"
        )
        
        # Wait for both results
        ingredient_result, nutrition_result = await asyncio.gather(
            ingredient_task, 
            nutrition_task
        )
        
        # Combine results and ensure JSON serialization
        combined_result = {
            "ingredient_detections": {
                "processing_time": ingredient_result.processing_time,
                "detections_count": len(ingredient_result.bboxes),
                "bboxes": ingredient_result.bboxes,
                "labels": ingredient_result.labels,
                "confidences": ingredient_result.confidences
            },
            "nutrition_detections": {
                "processing_time": nutrition_result.processing_time,
                "detections_count": len(nutrition_result.bboxes),
                "bboxes": nutrition_result.bboxes,
                "labels": nutrition_result.labels,
                "confidences": nutrition_result.confidences
            },
            "total_processing_time": ingredient_result.processing_time + nutrition_result.processing_time,
            "success": True
        }
        
        # Convert any remaining numpy types to JSON serializable types
        combined_result = convert_to_json_serializable(combined_result)
        
        return JSONResponse(content=combined_result)
        
    except Exception as e:
        logger.error(f"Error in combined detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Combined detection failed: {str(e)}")


async def _detect_json(
    file: UploadFile, 
    model_type: str, 
    confidence: float, 
    iou: float
) -> JSONResponse:
    """Internal function to handle JSON detection results"""
    
    # Validate file type
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Please upload JPG, JPEG, or PNG images."
        )
    
    # Validate model type
    available_models = get_available_models()
    if model_type not in available_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type. Available models: {available_models}"
        )
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Get model
        model = get_model(model_type)
        
        # Run detection in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            get_detection_results, 
            file_bytes, 
            model, 
            model_type
        )
        
        # Convert to response format and ensure JSON serialization
        response_data = {
            "model_type": result.model_type,
            "processing_time": result.processing_time,
            "detections_count": len(result.bboxes),
            "bboxes": result.bboxes,
            "labels": result.labels,
            "confidences": result.confidences,
            "success": True
        }
        
        # Convert any numpy types to JSON serializable types
        response_data = convert_to_json_serializable(response_data)
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Error in JSON detection: {str(e)}")
        response_data = {
            "model_type": model_type,
            "processing_time": 0.0,
            "detections_count": 0,
            "bboxes": [],
            "labels": [],
            "confidences": [],
            "success": False,
            "error_message": str(e)
        }
        return JSONResponse(content=response_data, status_code=500)


# Legacy endpoints for backward compatibility
@app.post("/detect/")
async def detect_legacy(
    file: UploadFile = File(...), 
    model_type: str = Query("ingredient", enum=["ingredient", "nutrition"])
):
    """
    Legacy detection endpoint for backward compatibility.
    
    Args:
        file: Image file
        model_type: Model type to use
        
    Returns:
        Annotated image
    """
    return await _detect_with_model(file, model_type, config.CONFIDENCE_THRESHOLD, config.IOU_THRESHOLD)


@app.get("/video_stream/{uid}")
async def video_stream(
    uid: str, 
    model_type: str = Query("ingredient", enum=["ingredient", "nutrition"])
):
    """
    Stream annotated video frames as MJPEG.
    
    Args:
        uid: Unique ID referencing the uploaded video
        model_type: Model type to use
        
    Returns:
        MJPEG stream of annotated frames
    """
    model = get_model(model_type)
    path = get_video_path(uid)
    
    if not path:
        raise HTTPException(status_code=404, detail="Video not found.")
    
    return StreamingResponse(
        generate_stream(path, model),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
