import os
import torch
from PIL import Image
from src.capture_photo import capture_photo, save_photo


def crop_box(img, left_ratio=0.0, top_ratio=0.0, right_ratio=0.5, bottom_ratio=1.0):
    """
    Crop a box from the image using ratio-based coordinates.
    
    Args:
        img: PIL Image object
        left_ratio: Left boundary as ratio of width (0.0 to 1.0), default 0.0
        top_ratio: Top boundary as ratio of height (0.0 to 1.0), default 0.0
        right_ratio: Right boundary as ratio of width (0.0 to 1.0), default 0.5 (left half)
        bottom_ratio: Bottom boundary as ratio of height (0.0 to 1.0), default 1.0
    
    Returns:
        PIL Image: Cropped image
        
    Examples:
        - Left half (default): left=0.0, right=0.5
        - Left third: left=0.0, right=0.33
        - Center third: left=0.33, right=0.67
        - Right half: left=0.5, right=1.0
    """
    width, height = img.size
    
    left = int(width * left_ratio)
    top = int(height * top_ratio)
    right = int(width * right_ratio)
    bottom = int(height * bottom_ratio)
    
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


def load_img(path_or_image, preprocess, save_processed_path=None):
    """
    Load, normalize brightness, and preprocess a single image.
    
    Args:
        path_or_image: Either a file path (str) or a PIL Image object
        preprocess: Image preprocessing transform
        save_processed_path: If provided, save processed image to this path
    
    Returns:
        tuple: (preprocessed_tensor, original_image)
    """
    # Handle both path and PIL Image input
    if isinstance(path_or_image, str):
        img = Image.open(path_or_image).convert("RGB")
    elif isinstance(path_or_image, Image.Image):
        img = path_or_image.convert("RGB")
    else:
        raise TypeError(f"Expected str or PIL.Image, got {type(path_or_image)}")
    
    # Crop the image (default: left half)
    # Adjust ratios in crop_box() call to change crop area
    # img = crop_box(img, left_ratio=0.0, top_ratio=0.2, right_ratio=0.5, bottom_ratio=0.95)
    
    if save_processed_path:
        img.save(save_processed_path)
        print(f"  Saved processed image: {save_processed_path}")
    return preprocess(img).unsqueeze(0), img



def list_images(folder):
    """List all image files in a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not os.path.exists(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in exts]


def predict(image, model, preprocess, prototypes, device, classes=None, temperature=100.0, save_processed_path=None):
    """
    Predict image class using the finetuned classifier.
    
    Args:
        image: Either a file path (str) or a PIL Image object
        model: Pre-loaded classifier model
        preprocess: Image preprocessing transform
        prototypes: Kept for compatibility, not used
        device: torch device (cpu/cuda/mps)
        classes: List of class names (default: ["open", "closed"])
        temperature: Temperature scaling factor for logits
        save_processed_path: If provided, save processed image to this path
    
    Returns:
        dict: Probabilities for each class
    """
    if classes is None:
        classes = ["open", "closed"]
    
    # Use sensible default temperature for classifier logits
    temp = 1.0 if temperature is None else float(temperature)
    with torch.no_grad():
        tensor, _ = load_img(image, preprocess, save_processed_path)
        logits = model(tensor.to(device))
        # Apply temperature scaling
        logits = logits / max(temp, 1e-6)
        probs = logits.softmax(dim=-1).squeeze(0).tolist()
    
    return {cls: prob for cls, prob in zip(classes, probs)}


def capture_and_predict(model, preprocess, prototypes, device, classes=None, save_photo_flag=False, camera=None):
    """
    Capture a photo from the camera and predict the class.
    Uses the existing capture_photo function to capture the image.
    
    Args:
        model: Pre-loaded CLIP model
        preprocess: Image preprocessing transform
        prototypes: Pre-computed class prototypes
        device: torch device (cpu/cuda/mps)
        classes: List of class names
        save_photo_flag: If True, save the captured photo to disk
        camera: Optional cv2.VideoCapture object
    
    Returns:
        tuple: (result_dict, photo) where result_dict contains probabilities and photo is the PIL Image object
               Returns (None, None) if capture failed
    """
    
    # Use existing capture_photo function (returns photo only)
    photo = capture_photo(camera)
    
    if not photo:
        print("Failed to capture photo")
        return None, None
    
    # Predict the result directly from the photo object
    result = predict(photo, model, preprocess, prototypes, device, classes)
    
    # Save photo to permanent location if requested
    if save_photo_flag:
        save_photo(photo)
    
    return result, photo
