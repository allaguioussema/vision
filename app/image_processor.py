import numpy as np
import cv2
from ultralytics import YOLO
import logging
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Data class for detection results"""
    bboxes: List[List[float]]  # [x1, y1, x2, y2, confidence, class_id]
    labels: List[str]
    confidences: List[float]
    processing_time: float
    model_type: str


class OptimizedImageProcessor:
    """Optimized image processor for ingredient and nutrition table detection"""
    
    def __init__(self):
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal model performance.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Ensure image is in BGR format (OpenCV default)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to BGR if needed
            if image.dtype == np.uint8:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def process_image(self, file_bytes: bytes, model: YOLO, model_type: str) -> bytes:
        """
        Process image with optimized detection pipeline.
        
        Args:
            file_bytes (bytes): Raw image data
            model (YOLO): YOLO model instance
            model_type (str): Type of model ("ingredient" or "nutrition")
            
        Returns:
            bytes: JPEG-encoded annotated image
        """
        start_time = time.time()
        
        try:
            # Convert bytes to OpenCV image
            img_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Preprocess image
            image = self.preprocess_image(image)
            
            # Run detection with optimized parameters
            results = model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                stream=False
            )
            
            processing_time = time.time() - start_time
            logger.info(f"{model_type} detection completed in {processing_time:.3f}s")
            
            # Draw annotations
            annotated = self.draw_detections(results[0], image, model_type)
            
            # Convert to JPEG with optimized quality
            _, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return encoded.tobytes()
            
        except Exception as e:
            logger.error(f"Error processing image with {model_type} model: {str(e)}")
            raise
    
    def draw_detections(self, result, image: np.ndarray, model_type: str) -> np.ndarray:
        """
        Draw detection results on image with custom styling.
        
        Args:
            result: YOLO detection result
            image (np.ndarray): Original image
            model_type (str): Type of model for custom styling
            
        Returns:
            np.ndarray: Annotated image
        """
        annotated = image.copy()
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            # Define colors for different model types
            colors = {
                "ingredient": (0, 255, 0),    # Green for ingredients
                "nutrition": (255, 0, 0)      # Red for nutrition tables
            }
            
            color = colors.get(model_type, (0, 255, 255))  # Default yellow
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{model_type}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Draw label background
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0], y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return annotated
    
    def get_detection_results(self, file_bytes: bytes, model: YOLO, model_type: str) -> DetectionResult:
        """
        Get structured detection results without drawing annotations.
        
        Args:
            file_bytes (bytes): Raw image data
            model (YOLO): YOLO model instance
            model_type (str): Type of model
            
        Returns:
            DetectionResult: Structured detection results
        """
        start_time = time.time()
        
        try:
            # Convert bytes to OpenCV image
            img_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Preprocess image
            image = self.preprocess_image(image)
            
            # Run detection
            results = model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                stream=False
            )
            
            processing_time = time.time() - start_time
            
            # Extract detection data
            bboxes = []
            labels = []
            confidences = []
            
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
                
                for box, conf, class_id in zip(boxes, confs, class_ids):
                    # Convert numpy values to Python native types for JSON serialization
                    bbox_list = [float(x) for x in box.tolist()] + [float(conf), int(class_id)]
                    bboxes.append(bbox_list)
                    labels.append(f"{model_type}_{int(class_id)}")
                    confidences.append(float(conf))
            
            return DetectionResult(
                bboxes=bboxes,
                labels=labels,
                confidences=confidences,
                processing_time=float(processing_time),
                model_type=model_type
            )
            
        except Exception as e:
            logger.error(f"Error getting detection results with {model_type} model: {str(e)}")
            raise


# Global processor instance
processor = OptimizedImageProcessor()


def process_image(file_bytes: bytes, model: YOLO, model_type: str) -> bytes:
    """
    Convenience function for backward compatibility.
    
    Args:
        file_bytes (bytes): Raw image data
        model (YOLO): YOLO model instance
        model_type (str): Type of model
        
    Returns:
        bytes: JPEG-encoded annotated image
    """
    return processor.process_image(file_bytes, model, model_type)


def get_detection_results(file_bytes: bytes, model: YOLO, model_type: str) -> DetectionResult:
    """
    Get structured detection results.
    
    Args:
        file_bytes (bytes): Raw image data
        model (YOLO): YOLO model instance
        model_type (str): Type of model
        
    Returns:
        DetectionResult: Structured detection results
    """
    return processor.get_detection_results(file_bytes, model, model_type)