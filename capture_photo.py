import cv2
import os
from datetime import datetime
import tempfile


def capture_photo():
    """
    Capture a photo using the system camera and save it to a temporary directory.
    The file is named with the current timestamp.
    
    Returns:
        str: The full path to the saved photo file, or None if capture failed.
    """
    # Initialize the camera (0 is usually the default camera)
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera")
        return None
    
    # Give the camera a moment to initialize
    import time
    time.sleep(0.5)
    
    # Capture a single frame
    ret, frame = camera.read()
    
    # Release the camera immediately
    camera.release()
    
    if not ret or frame is None:
        print("Error: Failed to capture image")
        return None
    
    # Get the system's temporary directory
    tmp_dir = os.getenv("DATA_DIR") + "/pics"

    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    # Create filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    
    # Full path to save the image
    filepath = os.path.join(tmp_dir, filename)
    
    # Save the captured image
    cv2.imwrite(filepath, frame)
    
    print(f"Photo saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Example usage
    photo_path = capture_photo()
    if photo_path:
        print(f"Successfully captured and saved photo: {photo_path}")
    else:
        print("Failed to capture photo")

