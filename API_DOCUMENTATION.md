# Custom Detection API Documentation

## Overview

The Custom Detection API is a high-performance FastAPI application that provides ingredient and nutrition table detection using custom YOLO models. This API replaces the previous YOLOv11 implementation with optimized custom models for superior performance and accuracy.

## Features

- **Dual Model Support**: Separate models for ingredient detection and nutrition table detection
- **High Performance**: Optimized inference pipeline with async processing
- **Comprehensive API**: Multiple endpoints for different use cases
- **Real-time Processing**: Support for both image and video processing
- **Robust Error Handling**: Comprehensive validation and error responses
- **Health Monitoring**: Built-in health checks and performance metrics
- **Testing Framework**: Comprehensive testing and benchmarking tools

## API Endpoints

### Health and Status

#### GET `/health`
Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "available_models": ["ingredient", "nutrition"],
  "uptime": 3600.5
}
```

#### GET `/models`
Get list of available models.

**Response:**
```json
["ingredient", "nutrition"]
```

### Detection Endpoints

#### POST `/detect/ingredient`
Detect ingredients in an uploaded image.

**Parameters:**
- `file`: Image file (JPG, PNG, JPEG)
- `confidence`: Confidence threshold (0.0-1.0, default: 0.5)
- `iou`: IoU threshold (0.0-1.0, default: 0.45)

**Response:** Annotated image as JPEG

#### POST `/detect/nutrition`
Detect nutrition tables in an uploaded image.

**Parameters:**
- `file`: Image file (JPG, PNG, JPEG)
- `confidence`: Confidence threshold (0.0-1.0, default: 0.5)
- `iou`: IoU threshold (0.0-1.0, default: 0.45)

**Response:** Annotated image as JPEG

#### POST `/detect/both`
Detect both ingredients and nutrition tables simultaneously.

**Parameters:**
- `file`: Image file (JPG, PNG, JPEG)
- `confidence`: Confidence threshold (0.0-1.0, default: 0.5)
- `iou`: IoU threshold (0.0-1.0, default: 0.45)

**Response:**
```json
{
  "ingredient_detections": {
    "processing_time": 0.125,
    "detections_count": 3,
    "bboxes": [[x1, y1, x2, y2, conf, class_id], ...],
    "labels": ["ingredient_0", "ingredient_1", "ingredient_2"],
    "confidences": [0.95, 0.87, 0.92]
  },
  "nutrition_detections": {
    "processing_time": 0.098,
    "detections_count": 1,
    "bboxes": [[x1, y1, x2, y2, conf, class_id]],
    "labels": ["nutrition_0"],
    "confidences": [0.89]
  },
  "total_processing_time": 0.223,
  "success": true
}
```

#### POST `/detect/json`
Get detection results as JSON without image annotations.

**Parameters:**
- `file`: Image file (JPG, PNG, JPEG)
- `model_type`: Model type ("ingredient" or "nutrition")
- `confidence`: Confidence threshold (0.0-1.0, default: 0.5)
- `iou`: IoU threshold (0.0-1.0, default: 0.45)

**Response:**
```json
{
  "model_type": "ingredient",
  "processing_time": 0.125,
  "detections_count": 3,
  "bboxes": [[x1, y1, x2, y2, conf, class_id], ...],
  "labels": ["ingredient_0", "ingredient_1", "ingredient_2"],
  "confidences": [0.95, 0.87, 0.92],
  "success": true
}
```

### Video Processing

#### POST `/detect/` (Legacy)
Legacy endpoint for video processing.

**Parameters:**
- `file`: Video file (MP4, AVI, MOV)
- `model_type`: Model type ("ingredient" or "nutrition")

**Response:** JSON with stream URL

#### GET `/video_stream/{uid}`
Stream annotated video frames.

**Parameters:**
- `uid`: Video unique ID
- `model_type`: Model type ("ingredient" or "nutrition")

**Response:** MJPEG stream

## Usage Examples

### Python Client Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Test health
response = requests.get(f"{BASE_URL}/health")
print("Health:", response.json())

# Detect ingredients
with open("food_image.jpg", "rb") as f:
    files = {"file": f}
    params = {"confidence": 0.6, "iou": 0.5}
    response = requests.post(f"{BASE_URL}/detect/ingredient", files=files, params=params)
    
    if response.status_code == 200:
        with open("detected_ingredients.jpg", "wb") as f:
            f.write(response.content)
        print("Ingredient detection completed")

# Detect both ingredients and nutrition tables
with open("food_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/detect/both", files=files)
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {results['ingredient_detections']['detections_count']} ingredients")
        print(f"Found {results['nutrition_detections']['detections_count']} nutrition tables")
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Detect ingredients
curl -X POST "http://localhost:8000/detect/ingredient" \
  -F "file=@food_image.jpg" \
  -F "confidence=0.6" \
  -F "iou=0.5" \
  --output detected_ingredients.jpg

# Get JSON results
curl -X POST "http://localhost:8000/detect/json" \
  -F "file=@food_image.jpg" \
  -F "model_type=ingredient" \
  -F "confidence=0.6"
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd YOLO-Object-Detection-App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure model files are in the `models/` directory:
   - `models/ingredient.pt`
   - `models/nutritiontable.pt`

### Running the Application

#### Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Using Gunicorn (Recommended for Production)
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Performance Optimization

### Model Optimization

1. **Model Quantization**: Consider quantizing models for faster inference
2. **Batch Processing**: Implement batch processing for multiple images
3. **GPU Optimization**: Ensure CUDA is properly configured
4. **Memory Management**: Monitor memory usage and optimize accordingly

### API Optimization

1. **Async Processing**: All detection operations are async for better throughput
2. **Thread Pool**: CPU-intensive tasks run in thread pool to avoid blocking
3. **Caching**: Models are cached in memory for faster access
4. **Connection Pooling**: Use connection pooling for database operations

### Configuration

Create a `config.py` file for environment-specific settings:

```python
import os

class Config:
    # Model settings
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
    
    # Performance settings
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10MB"))
    
    # Model paths
    INGREDIENT_MODEL_PATH = os.getenv("INGREDIENT_MODEL_PATH", "models/ingredient.pt")
    NUTRITION_MODEL_PATH = os.getenv("NUTRITION_MODEL_PATH", "models/nutritiontable.pt")
```

## Testing and Validation

### Running Tests

```bash
# Run comprehensive tests
python -m app.test_models

# Run specific test
python -c "from app.test_models import ModelTester; ModelTester().run_comprehensive_tests()"
```

### Benchmarking

The testing framework provides:
- **Performance Benchmarks**: Average inference time, throughput
- **Accuracy Testing**: Precision, recall, mAP metrics
- **Stress Testing**: Concurrent request handling
- **Memory Usage**: Resource consumption analysis

### Expected Performance

Based on typical hardware configurations:

| Hardware | Avg Inference Time | Throughput |
|----------|-------------------|------------|
| CPU (Intel i7) | 200-500ms | 2-5 req/s |
| GPU (RTX 3080) | 50-150ms | 10-20 req/s |
| GPU (V100) | 30-100ms | 15-30 req/s |

## Deployment Guidelines

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t custom-detection-api .
docker run -p 8000:8000 custom-detection-api
```

### Kubernetes Deployment

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: custom-detection-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: custom-detection-api
  template:
    metadata:
      labels:
        app: custom-detection-api
    spec:
      containers:
      - name: api
        image: custom-detection-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: MAX_WORKERS
          value: "4"
---
apiVersion: v1
kind: Service
metadata:
  name: custom-detection-api-service
spec:
  selector:
    app: custom-detection-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Production Considerations

1. **Load Balancing**: Use multiple instances behind a load balancer
2. **Monitoring**: Implement health checks and metrics collection
3. **Logging**: Configure structured logging for debugging
4. **Security**: Implement authentication and rate limiting
5. **Backup**: Regular backups of model files and configurations

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files exist in `models/` directory
   - Check file permissions
   - Verify model file integrity

2. **Memory Issues**
   - Reduce `MAX_WORKERS` in configuration
   - Monitor memory usage with `htop` or similar tools
   - Consider model quantization

3. **Performance Issues**
   - Check GPU availability and CUDA installation
   - Monitor CPU usage and adjust thread pool size
   - Optimize image preprocessing pipeline

4. **API Errors**
   - Check request format and file types
   - Verify confidence and IoU threshold values
   - Review error logs for specific issues

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Use the built-in health endpoint and add custom metrics:

```python
# Custom metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return {
        "requests_per_second": calculate_rps(),
        "average_response_time": calculate_avg_response_time(),
        "memory_usage": get_memory_usage(),
        "gpu_utilization": get_gpu_utilization()
    }
```

## Support and Contributing

For issues and feature requests, please create an issue in the repository. For contributions, please follow the standard pull request process.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 