import os
import time
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
    CLASSES = ["open", "closed"]  # maps to pos, neg
    SUPPORT_FOLDERS = ["fewshot/pos", "fewshot/neg"]  # pos=open, neg=closed
    
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
    
    prototypes = torch.cat(prototypes, dim=0)  # (C=2, D)
    
    return model, preprocess, prototypes, device


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
    model, preprocess, prototypes, device = load_model()
    print(f"Model loaded successfully on device: {device}")
    
    # Monitoring loop configuration
    check_interval = 3  # seconds between checks
    start_time = time.time()
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
            
            # Capture and predict
            result, photo_path = capture_and_predict(model, preprocess, prototypes, device)
            
            if not result:
                print("Failed to capture or predict, skipping...")
                time.sleep(check_interval)
                continue
            
            # Determine current door state
            new_state = "open" if result['open'] > result['closed'] else "closed"
            confidence = result[new_state]
            
            print(f"  State: {new_state.upper()} (confidence: {confidence:.2%})")
            print(f"  Open: {result['open']:.2%} | Closed: {result['closed']:.2%}")
            
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
            
            # Delete photo AFTER sending email (if confidence is high)
            if confidence >= 0.70:
                if photo_path and os.path.exists(photo_path):
                    os.remove(photo_path)
                    print(f"  High confidence ({confidence:.2%}) - photo deleted")
                    photo_path = None
            else:
                print(f"  Low confidence ({confidence:.2%}) - photo kept for review")
            
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

