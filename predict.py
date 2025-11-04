import os
import torch
import torch.nn.functional as F
from PIL import Image
from capture_photo import capture_photo


def load_img(path, preprocess):
    """Load and preprocess a single image."""
    return preprocess(Image.open(path).convert("RGB")).unsqueeze(0)


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
        batch = torch.cat([load_img(p, preprocess) for p in paths], dim=0).to(device)
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


def predict(image_path, model, preprocess, prototypes, device, temperature=100.0):
    """
    Predict whether a door is open or closed in an image using prototypical networks.
    
    Args:
        image_path: Path to the image file
        model: Pre-loaded CLIP model
        preprocess: Image preprocessing transform
        prototypes: Pre-computed class prototypes (tensor of shape [2, D])
        device: torch device (cpu/cuda/mps)
        temperature: Temperature scaling for logits (higher = more confident)
    
    Returns:
        dict: Probabilities for 'open' and 'closed'
    """
    with torch.no_grad():
        feat = encode_images([image_path], model, preprocess, device)  # (1, D)
        # Cosine similarity -> logit with temperature scaling
        logits = temperature * feat @ prototypes.T  # (1, C)
        probs = logits.softmax(dim=-1).squeeze(0).tolist()
    
    prob_open, prob_closed = probs[0], probs[1]
    return {"open": prob_open, "closed": prob_closed}


def capture_and_predict(model, preprocess, prototypes, device):
    """
    Capture a photo from the camera and predict whether the door is open or closed.
    Uses the existing capture_photo function to capture the image.
    
    Args:
        model: Pre-loaded CLIP model
        preprocess: Image preprocessing transform
        prototypes: Pre-computed class prototypes
        device: torch device (cpu/cuda/mps)
    
    Returns:
        tuple: (result_dict, photo_path) where result_dict contains probabilities and photo_path is the saved image path
               Returns (None, None) if capture failed
    """
    # Use existing capture_photo function
    photo_path = capture_photo()
    
    if not photo_path:
        print("Failed to capture photo")
        return None, None
    
    # Predict the result
    result = predict(photo_path, model, preprocess, prototypes, device)
    
    return result, photo_path
