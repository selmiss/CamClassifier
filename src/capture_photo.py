import cv2
import os
from datetime import datetime
import tempfile
from PIL import Image


def capture_photo(camera=None):
    """
    Capture a photo using the system camera.
    Does not save to disk - returns the image in memory.
    
    Args:
        camera: Optional cv2.VideoCapture object. If None, will open and close camera automatically.
    
    Returns:
        PIL.Image: The captured photo as a PIL Image object, or None if capture failed.
    """
    # Use provided camera or create a new one
    should_release = False
    if camera is None:
        camera = cv2.VideoCapture(0)
        should_release = True
        
        if not camera.isOpened():
            print("Error: Could not open camera")
            return None
        
        # Give the camera a moment to initialize
        import time
        time.sleep(0.5)
    
    # Capture a single frame
    ret, frame = camera.read()
    
    # Release the camera only if we created it
    if should_release:
        camera.release()
    
    if not ret or frame is None:
        print("Error: Failed to capture image")
        return None
    
    # Convert BGR (OpenCV) to RGB (PIL) and create PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = Image.fromarray(frame_rgb)
    
    print("Photo captured successfully")
    return photo


def save_photo(photo, save_dir=None, filename=None):
    """
    Save a photo to disk.
    
    Args:
        photo: PIL Image object to save
        save_dir: Directory to save the photo (default: DATA_DIR/bak or ./bak)
        filename: Filename to use (default: timestamp.jpg)
    
    Returns:
        str: Full path to the saved photo file
    """
    # Determine save directory
    if save_dir is None:
        data_dir = os.getenv("DATA_DIR")
        if data_dir:
            save_dir = os.path.join(data_dir, "bak")
        else:
            save_dir = "bak"
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create filename with current timestamp if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.jpg"
    
    # Full path to save the image
    filepath = os.path.join(save_dir, filename)
    
    # Save the image
    photo.save(filepath)
    
    print(f"Photo saved to: {filepath}")
    return filepath


if __name__ == "__main__":
    # Example usage
    photo = capture_photo()
    if photo:
        print(f"Successfully captured photo")
        print(f"Photo size: {photo.size}")
        
        # Save the photo if needed
        saved_path = save_photo(photo)
        print(f"Saved to: {saved_path}")
    else:
        print("Failed to capture photo")

