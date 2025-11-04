import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional


def send_email(
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    subject: str,
    body: str,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    attachments: Optional[List[str]] = None,
    html: bool = False
):
    """
    Send an email with optional attachments.
    
    Args:
        sender_email: Email address of the sender
        sender_password: Password or app password for the sender's email account
        recipient_email: Email address of the recipient
        subject: Subject line of the email
        body: Body content of the email
        smtp_server: SMTP server address (default: Gmail)
        smtp_port: SMTP server port (default: 587 for TLS)
        attachments: List of file paths to attach to the email
        html: If True, the body will be sent as HTML
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    try:
        # Create the email message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject
        
        # Attach the body text
        mime_type = "html" if html else "plain"
        message.attach(MIMEText(body, mime_type))
        
        # Attach files if provided
        if attachments:
            for filepath in attachments:
                if not os.path.exists(filepath):
                    print(f"Warning: Attachment file not found: {filepath}")
                    continue
                
                filename = os.path.basename(filepath)
                
                # Try to attach as image first (for common image formats)
                if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    with open(filepath, 'rb') as f:
                        img = MIMEImage(f.read())
                        img.add_header('Content-Disposition', 'attachment', filename=filename)
                        message.attach(img)
                else:
                    # Attach as generic binary file
                    with open(filepath, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header('Content-Disposition', f'attachment; filename={filename}')
                        message.attach(part)
        
        # Connect to the SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable TLS encryption
        
        # Login to the email account
        server.login(sender_email, sender_password)
        
        # Send the email
        server.send_message(message)
        
        # Close the connection
        server.quit()
        
        print(f"Email sent successfully to {recipient_email}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("Error: Authentication failed. Check your email and password.")
        print("For Gmail, you may need to use an App Password instead of your regular password.")
        return False
    except smtplib.SMTPException as e:
        print(f"SMTP error occurred: {e}")
        return False
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def send_email_with_photo(
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    subject: str,
    body: str,
    photo_path: str,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587
):
    """
    Convenience function to send an email with a photo attachment.
    
    Args:
        sender_email: Email address of the sender
        sender_password: Password or app password for the sender's email account
        recipient_email: Email address of the recipient
        subject: Subject line of the email
        body: Body content of the email
        photo_path: Path to the photo file to attach
        smtp_server: SMTP server address (default: Gmail)
        smtp_port: SMTP server port (default: 587 for TLS)
    
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    return send_email(
        sender_email=sender_email,
        sender_password=sender_password,
        recipient_email=recipient_email,
        subject=subject,
        body=body,
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        attachments=[photo_path]
    )


if __name__ == "__main__":
    # Example usage
    # NOTE: For Gmail, you need to:
    # 1. Enable 2-factor authentication
    # 2. Generate an App Password at https://myaccount.google.com/apppasswords
    # 3. Use the App Password instead of your regular password
    
    # Load credentials from environment variables
    # Run: source local.init_env.sh
    sender = os.getenv("SENDER_EMAIL", "your_email@gmail.com")
    password = os.getenv("SENDER_PASSWORD", "your_app_password")
    recipient = os.getenv("RECIPIENT_EMAIL", "recipient@example.com")
    
    # Simple text email
    send_email(
        sender_email=sender,
        sender_password=password,
        recipient_email=recipient,
        subject="Test Email",
        body="This is a test email sent from Python!"
    )
    
    # Email with attachment
    # send_email(
    #     sender_email=sender,
    #     sender_password=password,
    #     recipient_email=recipient,
    #     subject="Email with Photo",
    #     body="Please find the attached photo.",
    #     attachments=["/path/to/photo.jpg"]
    # )

