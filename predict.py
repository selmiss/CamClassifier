import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from capture_photo import capture_photo
import cv2


def preprocess_stable(img_pil, clip_limit=2.0, tile=8,
                      target_mean=0.5, lo=1, hi=99):
    """
    先白平衡(可选)→ LAB-CLAHE(对齐亮度/对比) → 自适应gamma(对齐全局曝光) → 百分位拉伸(抑制极暗/极亮)
    返回：RGB 的 np.uint8
    """
    # --- [可选] 灰世界白平衡（只做温和校色，避免过强）
    def gray_world(img):
        img = img.astype(np.float32)
        m = img.reshape(-1,3).mean(0)
        gray = m.mean() + 1e-6
        scale = gray / (m + 1e-6)
        img = np.clip(img * scale, 0, 255)
        return img.astype(np.uint8)

    img = np.array(img_pil.convert("RGB"))
    img = gray_world(img)  # 如颜色已经正常，可注释掉

    # --- LAB + CLAHE（只增强 L 通道，颜色不乱）
    lab  = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L,A,B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile,tile))
    L = clahe.apply(L)

    # --- 自适应 gamma：把平均亮度推到 target_mean（0~1）
    Lf = L.astype(np.float32) / 255.0
    cur = max(Lf.mean(), 1e-6)
    # 使 (cur ** gamma) → target_mean
    gamma = np.log(max(target_mean,1e-6)) / np.log(cur)
    Lg = np.clip(np.power(Lf, gamma), 0, 1)

    # --- 百分位拉伸：对齐对比度并抑制极端噪点
    lo_v, hi_v = np.percentile(Lg, [lo, hi])
    if hi_v > lo_v:
        Ls = np.clip((Lg - lo_v) / (hi_v - lo_v), 0, 1)
    else:
        Ls = Lg

    L8 = (Ls * 255.0 + 0.5).astype(np.uint8)
    out = cv2.merge([L8, A, B])
    out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def load_img(path, preprocess, save_processed_path=None):
    """Load, normalize brightness, and preprocess a single image."""
    img = Image.open(path).convert("RGB")
    # img = preprocess_stable(img)  # Normalize brightness
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


def predict(image_path, model, preprocess, prototypes, device, classes=None, temperature=100.0, save_processed_path=None):
    """
    Predict image class using prototypical networks.
    
    Args:
        image_path: Path to the image file
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
        tensor, processed_img = load_img(image_path, preprocess, save_processed_path)
        feat = model.encode_image(tensor.to(device))
        feat = F.normalize(feat, dim=-1)
        # Cosine similarity -> logit with temperature scaling
        logits = temperature * feat @ prototypes.T  # (1, C)
        probs = logits.softmax(dim=-1).squeeze(0).tolist()
    
    return {cls: prob for cls, prob in zip(classes, probs)}


def capture_and_predict(model, preprocess, prototypes, device, classes=None):
    """
    Capture a photo from the camera and predict the class.
    Uses the existing capture_photo function to capture the image.
    
    Args:
        model: Pre-loaded CLIP model
        preprocess: Image preprocessing transform
        prototypes: Pre-computed class prototypes
        device: torch device (cpu/cuda/mps)
        classes: List of class names
    
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
    result = predict(photo_path, model, preprocess, prototypes, device, classes)
    
    return result, photo_path
