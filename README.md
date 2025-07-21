# Custom Detection API

A high-performance FastAPI application for ingredient and nutrition table detection using custom YOLO models. This application replaces the previous YOLOv11 implementation with optimized custom models for superior performance and accuracy.

## 🚀 Features

- **Dual Model Support**: Separate optimized models for ingredient detection and nutrition table detection
- **High Performance**: Async processing with optimized inference pipeline
- **Comprehensive API**: Multiple endpoints for different use cases
- **Real-time Processing**: Support for both image and video processing
- **Robust Error Handling**: Comprehensive validation and error responses
- **Health Monitoring**: Built-in health checks and performance metrics
- **Testing Framework**: Comprehensive testing and benchmarking tools
- **Production Ready**: Optimized for deployment with Docker and Kubernetes

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM
- Custom YOLO models:
  - `models/ingredient.pt` (53.5MB)
  - `models/nutritiontable.pt` (51.2MB)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd YOLO-Object-Detection-App
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify model files:**
Ensure your custom models are in the `models/` directory:
```bash
ls -la models/
# Should show:
# ingredient.pt
# nutritiontable.pt
```

## 🚀 Quick Start

### Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Gunicorn (Recommended)
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## 🔧 API Endpoints

### Health and Status
- `GET /health` - Check API health status
- `GET /models` - Get available models

### Detection Endpoints
- `POST /detect/ingredient` - Detect ingredients in image
- `POST /detect/nutrition` - Detect nutrition tables in image
- `POST /detect/both` - Detect both ingredients and nutrition tables
- `POST /detect/json` - Get detection results as JSON
- `POST /detect/` - Legacy endpoint for video processing

### Video Processing
- `GET /video_stream/{uid}` - Stream annotated video frames

## 💡 Usage Examples

### Python Client
```python
import requests

# Detect ingredients
with open("food_image.jpg", "rb") as f:
    files = {"file": f}
    params = {"confidence": 0.6, "iou": 0.5}
    response = requests.post("http://localhost:8000/detect/ingredient", 
                           files=files, params=params)
    
    if response.status_code == 200:
        with open("detected_ingredients.jpg", "wb") as f:
            f.write(response.content)

# Detect both ingredients and nutrition tables
with open("food_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/detect/both", files=files)
    
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

## 🧪 Testing

### Run Tests
```bash
# Test the API
python test_app.py

# Run comprehensive model tests
python -m app.test_models
```

### Expected Performance

| Hardware | Avg Inference Time | Throughput |
|----------|-------------------|------------|
| CPU (Intel i7) | 200-500ms | 2-5 req/s |
| GPU (RTX 3080) | 50-150ms | 10-20 req/s |
| GPU (V100) | 30-100ms | 15-30 req/s |

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker build -t custom-detection-api .

# Run container
docker run -p 8000:8000 custom-detection-api
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MAX_WORKERS=4
```

## ☸️ Kubernetes Deployment

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
```

## 🔧 Configuration

### Environment Variables
```bash
# Model settings
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# Performance settings
MAX_WORKERS=4
MAX_FILE_SIZE=10MB

# Model paths
INGREDIENT_MODEL_PATH=models/ingredient.pt
NUTRITION_MODEL_PATH=models/nutritiontable.pt
```

## 📊 Monitoring

### Health Check
```bash
curl http://localhost:8000/health
```

### Custom Metrics
```python
# Add custom metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return {
        "requests_per_second": calculate_rps(),
        "average_response_time": calculate_avg_response_time(),
        "memory_usage": get_memory_usage(),
        "gpu_utilization": get_gpu_utilization()
    }
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files exist in `models/` directory
   - Check file permissions
   - Verify model file integrity

2. **Memory Issues**
   - Reduce `MAX_WORKERS` in configuration
   - Monitor memory usage
   - Consider model quantization

3. **Performance Issues**
   - Check GPU availability and CUDA installation
   - Monitor CPU usage and adjust thread pool size
   - Optimize image preprocessing pipeline

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

### Model Optimization
- Model quantization for faster inference
- Batch processing for multiple images
- GPU optimization with CUDA
- Memory management optimization

### API Optimization
- Async processing for better throughput
- Thread pool for CPU-intensive tasks
- Model caching in memory
- Connection pooling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For issues and feature requests, please create an issue in the repository.

## 🔄 Migration from YOLOv11

This application replaces the previous YOLOv11 implementation. Key changes:

- **Custom Models**: Replaced YOLOv11 models with specialized ingredient and nutrition table models
- **Optimized Architecture**: Improved performance with async processing and better resource management
- **Enhanced API**: More comprehensive endpoints with better error handling
- **Testing Framework**: Built-in testing and benchmarking tools
- **Production Ready**: Optimized for deployment with proper monitoring and scaling

For detailed migration guide, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md).
# vision
# vision
# vision
