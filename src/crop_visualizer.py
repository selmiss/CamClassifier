import os
from PIL import Image, ImageDraw, ImageFont
import sys

def crop_box(img, left_ratio=0, top_ratio=0, right_ratio=0.5, bottom_ratio=1.0):
    """
    Crop a box from the image using ratio-based coordinates.
    
    Args:
        img: PIL Image object
        left_ratio: Left boundary as ratio of width (0.0 to 1.0)
        top_ratio: Top boundary as ratio of height (0.0 to 1.0)
        right_ratio: Right boundary as ratio of width (0.0 to 1.0)
        bottom_ratio: Bottom boundary as ratio of height (0.0 to 1.0)
    
    Returns:
        PIL Image: Cropped image
    """
    width, height = img.size
    
    left = int(width * left_ratio)
    top = int(height * top_ratio)
    right = int(width * right_ratio)
    bottom = int(height * bottom_ratio)
    
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


def visualize_crop(image_path, left_ratio=0, top_ratio=0, right_ratio=0.5, bottom_ratio=1.0, output_dir="logs/crop_preview"):
    """
    Visualize the crop area on the original image and show the cropped result.
    
    Args:
        image_path: Path to the image file
        left_ratio: Left boundary as ratio of width (0.0 to 1.0)
        top_ratio: Top boundary as ratio of height (0.0 to 1.0)
        right_ratio: Right boundary as ratio of width (0.0 to 1.0)
        bottom_ratio: Bottom boundary as ratio of height (0.0 to 1.0)
        output_dir: Directory to save preview images
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Calculate crop box in pixels
    left = int(width * left_ratio)
    top = int(height * top_ratio)
    right = int(width * right_ratio)
    bottom = int(height * bottom_ratio)
    
    # Create visualization with crop box overlay
    img_with_box = img.copy()
    draw = ImageDraw.Draw(img_with_box)
    
    # Draw the crop box in red
    draw.rectangle([left, top, right, bottom], outline="red", width=5)
    
    # Add labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "RED BOX = CROP AREA", fill="red", font=font)
    draw.text((10, 50), f"Box: ({left}, {top}) to ({right}, {bottom})", fill="red", font=font)
    draw.text((10, 90), f"Crop size: {right-left}x{bottom-top}px", fill="red", font=font)
    
    # Get cropped image
    cropped = crop_box(img, left_ratio, top_ratio, right_ratio, bottom_ratio)
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Save both images
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
    cropped_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
    
    img_with_box.save(overlay_path, quality=95)
    cropped.save(cropped_path, quality=95)
    
    return overlay_path, cropped_path


def test_crop_on_samples(left_ratio=0, top_ratio=0, right_ratio=0.5, bottom_ratio=1.0):
    """
    Test the crop settings on sample images from fewshot folders.
    
    Args:
        left_ratio: Left boundary as ratio of width (0.0 to 1.0)
        top_ratio: Top boundary as ratio of height (0.0 to 1.0)
        right_ratio: Right boundary as ratio of width (0.0 to 1.0)
        bottom_ratio: Bottom boundary as ratio of height (0.0 to 1.0)
    """
    print("="*60)
    print("CROP BOX VISUALIZER")
    print("="*60)
    print(f"\nCrop Settings:")
    print(f"  Left:   {left_ratio*100:.1f}% of width")
    print(f"  Top:    {top_ratio*100:.1f}% of height")
    print(f"  Right:  {right_ratio*100:.1f}% of width")
    print(f"  Bottom: {bottom_ratio*100:.1f}% of height")
    print(f"\nCrop Area: {(right_ratio-left_ratio)*100:.1f}% × {(bottom_ratio-top_ratio)*100:.1f}%")
    print("\n" + "="*60)
    
    # Sample images from each class
    sample_folders = [
        ("fewshot/pos", "OPEN (pos)"),
        ("fewshot/neg", "CLOSED (neg)"),
        ("fewshot/neg_dark", "CLOSED DARK (neg_dark)")
    ]
    
    total_processed = 0
    
    for folder, label in sample_folders:
        if not os.path.exists(folder):
            print(f"\n⚠ Folder not found: {folder}")
            continue
        
        # Get first 2 images from each folder
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images = sorted(images)[:2]  # Take first 2
        
        if not images:
            print(f"\n⚠ No images found in: {folder}")
            continue
        
        print(f"\n{label}:")
        for img_name in images:
            img_path = os.path.join(folder, img_name)
            try:
                overlay_path, cropped_path = visualize_crop(
                    img_path, left_ratio, top_ratio, right_ratio, bottom_ratio
                )
                print(f"  ✓ {img_name}")
                print(f"    Overlay:  {overlay_path}")
                print(f"    Cropped:  {cropped_path}")
                total_processed += 1
            except Exception as e:
                print(f"  ✗ {img_name}: {e}")
    
    print("\n" + "="*60)
    print(f"Processed {total_processed} images")
    print("Check 'logs/crop_preview/' for results")
    print("="*60)


if __name__ == "__main__":
    # Default crop settings (currently left half)
    LEFT = 0.0    # 0% from left
    TOP = 0.0     # 0% from top
    RIGHT = 0.5   # 50% from left (left half)
    BOTTOM = 1.0  # 100% from top (full height)
    
    # You can adjust these values:
    # Examples:
    # - Left third: LEFT=0.0, RIGHT=0.33
    # - Center third: LEFT=0.33, RIGHT=0.67
    # - Right half: LEFT=0.5, RIGHT=1.0
    # - Top half: TOP=0.0, BOTTOM=0.5
    # - Center square: LEFT=0.25, TOP=0.25, RIGHT=0.75, BOTTOM=0.75
    
    if len(sys.argv) == 5:
        # Command line arguments: left top right bottom
        LEFT = float(sys.argv[1])
        TOP = float(sys.argv[2])
        RIGHT = float(sys.argv[3])
        BOTTOM = float(sys.argv[4])
    
    test_crop_on_samples(LEFT, TOP, RIGHT, BOTTOM)

