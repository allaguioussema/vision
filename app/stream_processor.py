import cv2
import tempfile
import uuid
from ultralytics import YOLO

video_registry = {}


def save_temp_video(file_bytes: bytes) -> str:
    """
    Save uploaded video to a temporary file and register it with a UUID.
    """
    uid = str(uuid.uuid4())
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4", prefix=f"{uid}_"
    )
    temp_file.write(file_bytes)
    temp_file.close()
    video_registry[uid] = temp_file.name
    return uid


def get_video_path(uid: str) -> str | None:
    """
    Return the path of the video file registered by UUID.
    """
    return video_registry.get(uid)


def generate_stream(video_path: str, model: YOLO):
    """
    Generator that yields annotated frames from the video as MJPEG stream.
    """
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated = results[0].plot()

        ret, buffer = cv2.imencode(".jpg", annotated)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
    cap.release()
