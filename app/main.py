from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import io

from .model_loader import load_models
from .image_processor import process_image
from .stream_processor import generate_stream, save_temp_video, get_video_path

# Load all YOLO models once at startup
models = load_models()

# Create FastAPI app
app = FastAPI(
    title="YOLO Detection API",
    description="Upload an image or video and get YOLO detection results.",
    version="1.0.0",
)

# Serve static files (e.g., CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 template engine setup
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Render the main upload form page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect/")
async def detect(
    file: UploadFile = File(...), model_type: str = Query("n", enum=["n", "s", "m"])
):
    """
    Detect objects in an uploaded image or video using YOLO.

    - If an image is uploaded: returns the annotated image as JPEG.
    - If a video is uploaded: returns a JSON with a stream URL.

    Args:
        file (UploadFile): Uploaded image or video.
        model_type (str): YOLO model variant to use.

    Returns:
        StreamingResponse or JSON: Annotated image or video stream link.
    """
    file_bytes = await file.read()
    filename = file.filename.lower()
    model = models.get(model_type)

    if filename.endswith((".jpg", ".jpeg", ".png")):
        try:
            result = process_image(file_bytes, model)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image.")
        return StreamingResponse(io.BytesIO(result), media_type="image/jpeg")

    elif filename.endswith((".mp4", ".avi", ".mov")):
        uid = save_temp_video(file_bytes)
        return {"stream_url": f"/video_stream/{uid}?model_type={model_type}"}

    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Please upload an image or video.",
    )


@app.get("/video_stream/{uid}")
async def video_stream(uid: str, model_type: str = Query("n", enum=["n", "s", "m"])):
    """
    Stream annotated video frames as MJPEG.

    Args:
        uid (str): Unique ID referencing the uploaded video.
        model_type (str): YOLO model variant to use.

    Returns:
        StreamingResponse: MJPEG stream of annotated frames.
    """
    model = models.get(model_type)
    path = get_video_path(uid)

    if not path:
        raise HTTPException(status_code=404, detail="Video not found.")

    return StreamingResponse(
        generate_stream(path, model),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
