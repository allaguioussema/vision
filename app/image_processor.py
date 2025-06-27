import numpy as np
import cv2
from ultralytics import YOLO


def process_image(file_bytes: bytes, model: YOLO) -> bytes:
    """
    Run YOLO on an image and return the annotated result as JPEG bytes.

    Args:
        file_bytes: Raw image data.
        model: YOLO model object.

    Returns:
        JPEG-encoded annotated image as bytes.
    """
    # Convert bytes to OpenCV image
    img_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # Run detection
    results = model(image)

    # Draw boxes on image
    annotated = results[0].plot()

    # Convert back to JPEG bytes
    _, encoded = cv2.imencode(".jpg", annotated)
    return encoded.tobytes()


#  Bytes Image -> Numpy Array -> OpenCV Image -> YOLO Model -> Annotated Image -> JPEG Bytes -> Return Bytes