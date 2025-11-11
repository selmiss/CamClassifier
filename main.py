import os
import sys
import time
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import cv2
from src.predict import predict, capture_and_predict
from src.send_email import send_email_with_photo
from src.capture_photo import save_photo


class RotatingLogger:
    """
    Logger that rotates log files every N hours and outputs to both console and file.
    """
    def __init__(self, log_dir="logs", rotation_hours=2):
        self.log_dir = log_dir
        self.rotation_hours = rotation_hours
        self.rotation_seconds = rotation_hours * 3600
        self.current_log_file = None
        self.log_file_handle = None
        self.log_start_time = None
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize first log file
        self._rotate_log_file()
    
    def _rotate_log_file(self):
        """Create a new log file with timestamp-based name."""
        # Close existing log file if open
        if self.log_file_handle:
            self.log_file_handle.close()
        
        # Generate new log filename based on current time
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.current_log_file = os.path.join(self.log_dir, f"monitoring_{timestamp}.log")
        self.log_start_time = time.time()
        
        # Open new log file
        self.log_file_handle = open(self.current_log_file, 'a', buffering=1)  # Line buffering
        
        # Write header
        header = f"\n{'='*70}\n"
        header += f"Log started at: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Log file: {self.current_log_file}\n"
        header += f"{'='*70}\n"
        self.log_file_handle.write(header)
        self.log_file_handle.flush()
    
    def _should_rotate(self):
        """Check if it's time to rotate the log file."""
        if self.log_start_time is None:
            return True
        elapsed = time.time() - self.log_start_time
        return elapsed >= self.rotation_seconds
    
    def log(self, message, console_only=False):
        """
        Log a message with timestamp to both console and file.
        
        Args:
            message: Message to log
            console_only: If True, only print to console, not to file
        """
        # Check if we need to rotate
        if not console_only and self._should_rotate():
            self._rotate_log_file()
        
        # Add timestamp to message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamped_msg = f"[{timestamp}] {message}"
        
        # Print to console
        print(timestamped_msg)
        
        # Write to file (unless console_only)
        if not console_only and self.log_file_handle:
            self.log_file_handle.write(timestamped_msg + '\n')
            self.log_file_handle.flush()
    
    def log_raw(self, message, console_only=False):
        """
        Log a message without timestamp (for formatting like separators).
        
        Args:
            message: Message to log
            console_only: If True, only print to console, not to file
        """
        # Check if we need to rotate
        if not console_only and self._should_rotate():
            self._rotate_log_file()
        
        # Print to console
        print(message)
        
        # Write to file (unless console_only)
        if not console_only and self.log_file_handle:
            self.log_file_handle.write(message + '\n')
            self.log_file_handle.flush()
    
    def close(self):
        """Close the log file."""
        if self.log_file_handle:
            footer = f"\n{'='*70}\n"
            footer += f"Log ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            footer += f"{'='*70}\n"
            self.log_file_handle.write(footer)
            self.log_file_handle.close()
            self.log_file_handle = None


def load_model():
    """
    Load a small pretrained model and restore weights from checkpoint for inference.
    
    Returns:
        tuple: (model, preprocess, prototypes, classes, device)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Initialize pretrained backbone and preprocessing
    weights = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    model = mobilenet_v3_small(weights=weights)
    
    # Load checkpoint path from env or default
    ckpt_path = os.getenv("CHECKPOINT_PATH", os.path.join("checkpoints", "mobilenet_v3_small_fewshot.pt"))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'. Train first with: python src/train.py")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Read classes from checkpoint (fallback to two-class order if missing)
    classes = ckpt.get("classes", ["open", "closed"])
    
    # Replace classifier head to match number of classes and load weights
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(classes))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device).eval()
    
    # API compatibility: prototypes unused
    prototypes = None
    return model, preprocess, prototypes, classes, device


def main():
    """
    Main function to continuously monitor door state and send email alerts on state changes.
    Runs for 30 seconds, capturing photos every few seconds.
    """
    # Initialize logger (rotates every 2 hours)
    logger = RotatingLogger(log_dir="logs", rotation_hours=2)
    
    # Load email credentials from environment variables
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL")
    
    if not all([sender_email, sender_password, recipient_email]):
        logger.log("Warning: Email credentials not set. Please run 'source local.env.sh'")
        logger.log("Set SENDER_EMAIL, SENDER_PASSWORD, and RECIPIENT_EMAIL environment variables.")
        logger.log("Continuing without email alerts...")
        logger.log_raw("")
        email_enabled = False
    else:
        email_enabled = True
        logger.log(f"Email alerts enabled. Will send to: {recipient_email}")
        logger.log_raw("")
    
    # Load model and prepare visual prototypes from few-shot examples
    logger.log("Loading model and computing prototypes...")
    model, preprocess, prototypes, classes, device = load_model()
    logger.log(f"Model loaded successfully on device: {device}")
    logger.log(f"Classes: {classes}")
    
    # Monitoring loop configuration
    check_interval = 30  # seconds between checks
    model_reload_interval = 3600  # seconds (1 hour)
    start_time = time.time()
    last_model_load_time = time.time()
    current_state = None
    iteration = 0
    
    logger.log_raw("\n" + "="*50)
    logger.log("Starting door monitoring (running forever)")
    logger.log("Press Ctrl+C to stop")
    logger.log_raw("="*50)
    
    # Open camera once at the beginning
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.log("Error: Could not open camera")
        logger.close()
        return
    
    # Give the camera a moment to initialize
    time.sleep(0.5)
    logger.log("Camera initialized successfully")
    logger.log_raw("")
    
    try:
        while True:
            iteration += 1
            elapsed = time.time() - start_time
            logger.log_raw(f"\n[Check #{iteration} @ {elapsed:.1f}s]")
            
            # Check if it's time to reload the model (every hour) 
            # ================================================================================================================
            # time_since_last_load = time.time() - last_model_load_time
            # if time_since_last_load >= model_reload_interval:
            #     logger.log(f"  Reloading model (last loaded {time_since_last_load/60:.1f} minutes ago)...")
            #     try:
            #         model, preprocess, prototypes, classes, device = load_model()
            #         last_model_load_time = time.time()
            #         logger.log(f"  ✓ Model reloaded successfully")
            #     except Exception as e:
            #         logger.log(f"  ✗ Failed to reload model: {e}")
            #         logger.log(f"  Continuing with existing model...")
            # ================================================================================================================
            
            # Check if current time is in quiet hours (7pm to 8am)
            current_hour = datetime.now().hour
            if 19 <= current_hour or current_hour < 9:
                logger.log(f"  Quiet hours (7pm-8am) - skipping capture (current time: {datetime.now().strftime('%H:%M:%S')})")
                time.sleep(check_interval)
                continue
            
            # Capture and predict (photo is in memory, not saved yet)
            result, photo = capture_and_predict(model, preprocess, prototypes, device, classes, camera=camera)
            
            if not result:
                logger.log("Failed to capture or predict, skipping...")
                time.sleep(check_interval)
                continue
            
            # Determine current door state
            # Combine closed and closed_dark as "closed" state
            closed_total = result['closed'] + result.get('closed_dark', 0)
            new_state = "open" if result['open'] > closed_total else "closed"
            confidence = result['open'] if new_state == "open" else closed_total
            
            logger.log(f"  State: {new_state.upper()} (confidence: {confidence:.2%})")
            prob_str = " | ".join([f"{cls.capitalize()}: {result[cls]:.2%}" for cls in classes])
            logger.log(f"  {prob_str}")
            
            # Determine if we need to save the photo
            photo_path = None
            state_changed = False
            
            # Check for state change
            if current_state is None:
                # First check - just initialize state
                current_state = new_state
                logger.log(f"  Initial state set to: {current_state.upper()}")
            elif current_state != new_state:
                # State changed!
                state_changed = True
                logger.log_raw("")
                logger.log(f"🚨 STATE CHANGE DETECTED: {current_state.upper()} → {new_state.upper()}")
                
                # Save photo for email when state changes
                
                photo_path = save_photo(photo)
                logger.log(f"  Photo saved for state change alert: {photo_path}")
                
                if email_enabled:
                    subject = f"Door Message: {new_state.upper()} Detected"
                    body = f"""Door state change detected!

Previous State: {current_state.upper()}
New State: {new_state.upper()}
Confidence: {confidence:.2%}

Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Photo attached.
"""
                    
                    logger.log(f"  Sending email to {recipient_email}...")
                    
                    # Send email with photo
                    success = send_email_with_photo(
                        sender_email=sender_email,
                        sender_password=sender_password,
                        recipient_email=recipient_email,
                        subject=subject,
                        body=body,
                        photo_path=photo_path
                    )
                    
                    if success:
                        logger.log("  ✓ Email sent successfully!")
                    else:
                        logger.log("  ✗ Failed to send email")
                else:
                    logger.log("  (Email alerts disabled)")
                
                # Update state
                current_state = new_state
            else:
                logger.log(f"  State unchanged: {current_state.upper()}")
            
            # Handle photo based on confidence
            if confidence >= 0.6:
                # High confidence - delete saved photo if it exists
                if photo_path and os.path.exists(photo_path):
                    os.remove(photo_path)
                    logger.log(f"  High confidence ({confidence:.2%}) - photo deleted (email already sent)")
                else:
                    logger.log(f"  High confidence ({confidence:.2%}) - photo not saved")
            else:
                # Low confidence - save for review if not already saved
                if not photo_path:
                    photo_path = save_photo(photo)
                    logger.log(f"  Low confidence ({confidence:.2%}) - photo saved for review: {photo_path}")
                else:
                    logger.log(f"  Low confidence ({confidence:.2%}) - photo kept for review: {photo_path}")
            
            # Wait before next check
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        logger.log_raw("\n")
        logger.log("Monitoring interrupted by user.")
    finally:
        # Close camera
        if camera is not None and camera.isOpened():
            camera.release()
            logger.log("Camera released")
        
        # Close logger
        logger.log_raw("\n" + "="*50)
        logger.log("Monitoring complete")
        logger.log(f"Final state: {current_state.upper() if current_state else 'UNKNOWN'}")
        logger.log_raw("="*50)
        logger.close()


if __name__ == "__main__":
    main()

