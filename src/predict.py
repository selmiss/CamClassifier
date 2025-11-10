import os
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from src.capture_photo import capture_photo, save_photo


# Default crop parameters used in preprocessing
CROP_LEFT_RATIO = 0.0
CROP_TOP_RATIO = 0.2
CROP_RIGHT_RATIO = 0.5
CROP_BOTTOM_RATIO = 0.95


def crop_box(img, left_ratio=0.0, top_ratio=0.0, right_ratio=0.5, bottom_ratio=1.0, return_coords=False):
    """
    Crop a box from the image using ratio-based coordinates.
    
    Args:
        img: PIL Image object
        left_ratio: Left boundary as ratio of width (0.0 to 1.0), default 0.0
        top_ratio: Top boundary as ratio of height (0.0 to 1.0), default 0.0
        right_ratio: Right boundary as ratio of width (0.0 to 1.0), default 0.5 (left half)
        bottom_ratio: Bottom boundary as ratio of height (0.0 to 1.0), default 1.0
        return_coords: If True, return (cropped_img, crop_coords) instead of just cropped_img
    
    Returns:
        PIL Image: Cropped image (if return_coords=False)
        tuple: (cropped_img, crop_coords) where crop_coords is a dict with 'left', 'top', 'right', 'bottom' (if return_coords=True)
        
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
    
    if return_coords:
        crop_coords = {
            'left': left,
            'top': top,
            'right': right,
            'bottom': bottom,
            'left_ratio': left_ratio,
            'top_ratio': top_ratio,
            'right_ratio': right_ratio,
            'bottom_ratio': bottom_ratio
        }
        return cropped_img, crop_coords
    
    return cropped_img


def apply_edge_detection(img, method='scharr'):
    """
    Apply edge detection to an image using Sobel operator with adaptive thresholding.
    
    Args:
        img: PIL Image object
        method: Edge detection method ('sobel', 'canny', or 'scharr')
    
    Returns:
        PIL Image: Edge-detected image in RGB format (edges are white on black background)
    """
    # Convert PIL Image to numpy array
    img_np = np.array(img)
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    if method == 'sobel':
        # Compute Sobel gradients in X and Y directions
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-255 range
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        # Apply adaptive thresholding to get cleaner edges
        _, edges = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif method == 'scharr':
        # Scharr operator (more accurate than Sobel)
        scharrx = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)
        
        # Compute gradient magnitude
        magnitude = np.sqrt(scharrx**2 + scharry**2)
        
        # Normalize to 0-255 range
        magnitude = np.uint8(255 * magnitude / np.max(magnitude))
        
        # Apply adaptive thresholding
        _, edges = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    else:  # canny (fallback)
        edges = cv2.Canny(blurred, 50, 150)
    
    # Convert single channel edge map to 3-channel RGB (edges are white on black)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Convert back to PIL Image
    return Image.fromarray(edges_rgb)


def load_img(path_or_image, preprocess, save_processed_path=None):
    """
    Load, crop, apply edge detection, and preprocess a single image.
    
    Args:
        path_or_image: Either a file path (str) or a PIL Image object
        preprocess: Image preprocessing transform
        save_processed_path: If provided, save processed image to this path
    
    Returns:
        tuple: (preprocessed_tensor, processed_image)
    """
    # Handle both path and PIL Image input
    if isinstance(path_or_image, str):
        img = Image.open(path_or_image).convert("RGB")
    elif isinstance(path_or_image, Image.Image):
        img = path_or_image.convert("RGB")
    else:
        raise TypeError(f"Expected str or PIL.Image, got {type(path_or_image)}")
    
    # Crop the image using default crop parameters
    img = crop_box(img, left_ratio=CROP_LEFT_RATIO, top_ratio=CROP_TOP_RATIO, 
                   right_ratio=CROP_RIGHT_RATIO, bottom_ratio=CROP_BOTTOM_RATIO)
    
    # Apply edge detection after cropping
    img = apply_edge_detection(img)
    
    if save_processed_path:
        img.save(save_processed_path)
        print(f"  Saved processed image: {save_processed_path}")
    return preprocess(img).unsqueeze(0), img


def encode_images(paths, model, preprocess, device):
    """
    Encode a batch of images to feature vectors.
    
    Args:
        paths: List of image paths
        model: CLIP model
        preprocess: Image preprocessing transform
        device: torch device
    
    Returns:
        torch.Tensor: Normalized feature vectors (N, D)
    """
    with torch.no_grad():
        tensors = []
        for p in paths:
            tensor, _ = load_img(p, preprocess)
            tensors.append(tensor)
        batch = torch.cat(tensors, dim=0).to(device)
        feats = model.encode_image(batch)
        feats = F.normalize(feats, dim=-1)  # Normalize for cosine similarity
    return feats


def list_images(folder):
    """List all image files in a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not os.path.exists(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in exts]


def predict(image, model, preprocess, prototypes, device, classes=None, temperature=100.0, save_processed_path=None):
    """
    Predict image class using prototypical networks.
    
    Args:
        image: Either a file path (str) or a PIL Image object
        model: Pre-loaded CLIP model
        preprocess: Image preprocessing transform
        prototypes: Pre-computed class prototypes (tensor of shape [C, D])
        device: torch device (cpu/cuda/mps)
        classes: List of class names (default: ["open", "closed"])
        temperature: Temperature scaling for logits (higher = more confident)
        save_processed_path: If provided, save processed image to this path
    
    Returns:
        dict: Probabilities for each class
    """
    if classes is None:
        classes = ["open", "closed"]
    
    with torch.no_grad():
        tensor, processed_img = load_img(image, preprocess, save_processed_path)
        feat = model.encode_image(tensor.to(device))
        feat = F.normalize(feat, dim=-1)
        # Cosine similarity -> logit with temperature scaling
        logits = temperature * feat @ prototypes.T  # (1, C)
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
