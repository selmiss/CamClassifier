"""
Example script that combines photo capture and email sending.
Captures a photo from the camera and sends it via email.
"""

from capture_photo import capture_photo
from send_email import send_email_with_photo


def capture_and_send_photo(
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    subject: str = "Photo from Camera",
    body: str = "Please find the attached photo captured from my camera."
):
    """
    Capture a photo from the camera and send it via email.
    
    Args:
        sender_email: Email address of the sender
        sender_password: Password or app password for the sender's email account
        recipient_email: Email address of the recipient
        subject: Subject line of the email
        body: Body content of the email
    
    Returns:
        bool: True if photo was captured and sent successfully, False otherwise
    """
    # Capture the photo
    print("Capturing photo from camera...")
    photo_path = capture_photo()
    
    if not photo_path:
        print("Failed to capture photo. Aborting.")
        return False
    
    # Send the email with the photo
    print(f"Sending email to {recipient_email}...")
    success = send_email_with_photo(
        sender_email=sender_email,
        sender_password=sender_password,
        recipient_email=recipient_email,
        subject=subject,
        body=body,
        photo_path=photo_path
    )
    
    return success


if __name__ == "__main__":
    # Example usage
    # IMPORTANT: Configure these with your actual email credentials
    # Load credentials from environment variables
    # Run: source local.init_env.sh
    import os
    
    SENDER_EMAIL = os.getenv("SENDER_EMAIL", "your_email@gmail.com")
    SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "your_app_password")  # Use App Password for Gmail
    RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "recipient@example.com")
    
    print("=" * 50)
    print("Capture and Send Photo")
    print("=" * 50)
    
    success = capture_and_send_photo(
        sender_email=SENDER_EMAIL,
        sender_password=SENDER_PASSWORD,
        recipient_email=RECIPIENT_EMAIL,
        subject="Camera Photo",
        body="Here's a photo I just captured!"
    )
    
    if success:
        print("\n✓ Photo captured and sent successfully!")
    else:
        print("\n✗ Failed to capture or send photo.")

