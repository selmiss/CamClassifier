import os
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import open_clip
from predict import predict, capture_and_predict, encode_images, list_images
from send_email import send_email_with_photo


def load_model():
    """
    Load CLIP model and prepare visual prototypes from few-shot examples.
    
    Returns:
        tuple: (model, preprocess, prototypes, device)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load CLIP model
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model = model.to(device).eval()
    
    # Define classes (order matters!)
    CLASSES = ["open", "closed", "closed_dark"]  # maps to pos, neg, neg_dark
    SUPPORT_FOLDERS = ["fewshot/pos", "fewshot/neg", "fewshot/neg_dark"]  # pos=open, neg=closed, neg_dark=closed_dark
    
    # Compute prototypes (mean features) for each class
    prototypes = []
    for cls, folder in zip(CLASSES, SUPPORT_FOLDERS):
        img_paths = list_images(folder)
        if not img_paths:
            raise ValueError(f"No images found in {folder}. Please add example images for '{cls}' class.")
        
        print(f"Loading {len(img_paths)} example(s) for class '{cls}' from {folder}")
        feats = encode_images(img_paths, model, preprocess, device)  # (Ns, D)
        proto = feats.mean(dim=0, keepdim=True)  # (1, D)
        proto = F.normalize(proto, dim=-1)  # Re-normalize for stability
        prototypes.append(proto)
    
    prototypes = torch.cat(prototypes, dim=0)  # (C=3, D)
    
    return model, preprocess, prototypes, CLASSES, device


def main():
    """
    Main function to continuously monitor door state and send email alerts on state changes.
    Runs for 30 seconds, capturing photos every few seconds.
    """
    # Load email credentials from environment variables
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL")
    
    if not all([sender_email, sender_password, recipient_email]):
        print("Warning: Email credentials not set. Please run 'source local.env.sh'")
        print("Set SENDER_EMAIL, SENDER_PASSWORD, and RECIPIENT_EMAIL environment variables.")
        print("Continuing without email alerts...\n")
        email_enabled = False
    else:
        email_enabled = True
        print(f"Email alerts enabled. Will send to: {recipient_email}\n")
    
    # Load model and prepare visual prototypes from few-shot examples
    print("Loading model and computing prototypes...")
    model, preprocess, prototypes, classes, device = load_model()
    print(f"Model loaded successfully on device: {device}")
    print(f"Classes: {classes}")
    
    # Monitoring loop configuration
    check_interval = 5  # seconds between checks
    model_reload_interval = 3600  # seconds (1 hour)
    start_time = time.time()
    last_model_load_time = time.time()
    current_state = None
    iteration = 0
    
    print("\n" + "="*50)
    print("Starting door monitoring (running forever)")
    print("Press Ctrl+C to stop")
    print("="*50)
    
    try:
        while True:
            iteration += 1
            elapsed = time.time() - start_time
            print(f"\n[Check #{iteration} @ {elapsed:.1f}s]")
            
            # Check if it's time to reload the model (every hour)
            time_since_last_load = time.time() - last_model_load_time
            if time_since_last_load >= model_reload_interval:
                print(f"  Reloading model (last loaded {time_since_last_load/60:.1f} minutes ago)...")
                try:
                    model, preprocess, prototypes, classes, device = load_model()
                    last_model_load_time = time.time()
                    print(f"  ✓ Model reloaded successfully")
                except Exception as e:
                    print(f"  ✗ Failed to reload model: {e}")
                    print(f"  Continuing with existing model...")
            
            # Check if current time is in quiet hours (7pm to 8am)
            current_hour = datetime.now().hour
            if 19 <= current_hour or current_hour < 8:
                print(f"  Quiet hours (7pm-8am) - skipping capture (current time: {datetime.now().strftime('%H:%M:%S')})")
                time.sleep(check_interval)
                continue
            
            # Capture and predict
            result, photo_path = capture_and_predict(model, preprocess, prototypes, device, classes)
            
            if not result:
                print("Failed to capture or predict, skipping...")
                time.sleep(check_interval)
                continue
            
            # Determine current door state
            # Combine closed and closed_dark as "closed" state
            closed_total = result['closed'] + result.get('closed_dark', 0)
            new_state = "open" if result['open'] > closed_total else "closed"
            confidence = result['open'] if new_state == "open" else closed_total
            
            print(f"  State: {new_state.upper()} (confidence: {confidence:.2%})")
            prob_str = " | ".join([f"{cls.capitalize()}: {result[cls]:.2%}" for cls in classes])
            print(f"  {prob_str}")
            
            # Check for state change FIRST (before deleting photo)
            state_changed = False
            if current_state is None:
                # First check - just initialize state
                current_state = new_state
                print(f"  Initial state set to: {current_state.upper()}")
            elif current_state != new_state:
                # State changed!
                state_changed = True
                print(f"\n🚨 STATE CHANGE DETECTED: {current_state.upper()} → {new_state.upper()}")
                
                if email_enabled:
                    subject = f"Door Alert: Changed from {current_state.upper()} to {new_state.upper()}"
                    body = f"""Door state change detected!

Previous State: {current_state.upper()}
New State: {new_state.upper()}
Confidence: {confidence:.2%}

Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Photo attached.
"""
                    
                    print(f"  Sending email to {recipient_email}...")
                    
                    # Always send email with photo when state changes
                    if photo_path and os.path.exists(photo_path):
                        success = send_email_with_photo(
                            sender_email=sender_email,
                            sender_password=sender_password,
                            recipient_email=recipient_email,
                            subject=subject,
                            body=body,
                            photo_path=photo_path
                        )
                    else:
                        from send_email import send_email
                        success = send_email(
                            sender_email=sender_email,
                            sender_password=sender_password,
                            recipient_email=recipient_email,
                            subject=subject,
                            body=body
                        )
                    
                    if success:
                        print("  ✓ Email sent successfully!")
                    else:
                        print("  ✗ Failed to send email")
                else:
                    print("  (Email alerts disabled)")
                
                # Update state
                current_state = new_state
            else:
                print(f"  State unchanged: {current_state.upper()}")
            
            # Handle photo based on confidence
            if confidence >= 0.70:
                # High confidence - delete photo
                if photo_path and os.path.exists(photo_path):
                    os.remove(photo_path)
                    print(f"  High confidence ({confidence:.2%}) - photo deleted")
                    photo_path = None
            else:
                # Low confidence - overwrite with processed version for review
                if photo_path and os.path.exists(photo_path):
                    # Re-process and save the processed image to original path
                    from predict import load_img
                    tensor, processed_img = load_img(photo_path, preprocess)
                    processed_img.save(photo_path)
                    print(f"  Low confidence ({confidence:.2%}) - overwritten with processed image for review")
            
            # Wait before next check
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user.")
    
    print("\n" + "="*50)
    print("Monitoring complete")
    print(f"Final state: {current_state.upper() if current_state else 'UNKNOWN'}")
    print("="*50)


if __name__ == "__main__":
    main()

