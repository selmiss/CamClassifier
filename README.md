# CamClassifier

A real-time visual monitoring system that uses CLIP-based few-shot learning to classify images from your camera and sends email alerts when changes are detected.

## Features

- 🎥 **Camera Integration** - Capture photos from system camera
- 🤖 **Visual Prototypical Networks** - Few-shot learning using CLIP image embeddings
- 🔄 **Continuous Monitoring** - Real-time state tracking with configurable intervals
- 📧 **Email Alerts** - Automatic notifications with photos when state changes
- ⚡ **Apple Silicon Optimized** - GPU acceleration on M1/M2/M3 using MPS backend

## How It Works

Instead of text-based classification, this system learns from **visual examples**:

1. **Training**: Computes prototype features by averaging CLIP embeddings from example images
2. **Monitoring**: Continuously captures photos and compares them to prototypes using cosine similarity
3. **Detection**: Detects state changes (e.g., door open ↔ closed)
4. **Alerting**: Sends email with the photo when a state change occurs

## Installation

### Step 1: Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note for macOS with Apple Silicon (M1/M2/M3):**
PyTorch will automatically use the MPS (Metal Performance Shaders) backend for GPU acceleration.

### Step 3: Prepare Few-Shot Examples

Create example images for your classification task:

```
fewshot/
├── pos/          # Examples of "open" state
│   ├── example1.jpg
│   ├── example2.jpg
│   └── example3.jpg
└── neg/          # Examples of "closed" state
    └── example1.jpg
```

You need at least 1 image per class. More examples generally improve accuracy.

### Step 4: Configuration

Set up your email credentials in `local.env.sh`:

```bash
export DATA_DIR="/path/to/your/project"
export SENDER_EMAIL="your_email@gmail.com"
export SENDER_PASSWORD="your_app_password"
export RECIPIENT_EMAIL="recipient@example.com"
```

Then source the environment file:
```bash
source local.env.sh
```

## Usage

### Main Monitoring System

Run continuous monitoring (checks every 3 seconds, runs forever):

```bash
python main.py
```

This will:
1. Load CLIP model and compute prototypes from your examples
2. Continuously capture photos from camera
3. Classify each photo as "open" or "closed"
4. Send email alert when state changes

**Output:**
```
Loading model and computing prototypes...
Loading 3 example(s) for class 'open' from fewshot/pos
Loading 1 example(s) for class 'closed' from fewshot/neg
Model loaded successfully on device: mps

==================================================
Starting door monitoring (running forever)
Press Ctrl+C to stop
==================================================

[Check #1 @ 0.0s]
  State: OPEN (confidence: 87.34%)
  Initial state set to: OPEN

[Check #2 @ 3.2s]
  State: CLOSED (confidence: 92.45%)

🚨 STATE CHANGE DETECTED: OPEN → CLOSED
  Sending email to user@example.com...
  ✓ Email sent successfully!
```

### Individual Components

#### 1. Capture Photo from Camera

```python
from capture_photo import capture_photo

photo_path = capture_photo()
print(f"Photo saved at: {photo_path}")
```

#### 2. Predict Image Classification

```python
from main import load_model
from predict import predict

# Load model and prototypes
model, preprocess, prototypes, device = load_model()

# Predict single image
result = predict("test.jpg", model, preprocess, prototypes, device)
print(result)  # {'open': 0.95, 'closed': 0.05}
```

#### 3. Send Email with Attachments

```python
from send_email import send_email_with_photo

send_email_with_photo(
    sender_email="your_email@gmail.com",
    sender_password="your_app_password",
    recipient_email="recipient@example.com",
    subject="Door Alert",
    body="Door state changed!",
    photo_path="photo.jpg"
)
```

## Technical Details

### Model Architecture
- **Vision Model**: OpenAI CLIP ViT-B-16
- **Classification Method**: Prototypical networks with cosine similarity
- **Temperature Scaling**: 100.0 (configurable for confidence adjustment)
- **Hardware Acceleration**: MPS backend on Apple Silicon

### Classification Process
1. Extract CLIP features from example images: `f_i ∈ ℝ^D`
2. Compute class prototypes: `c_k = normalize(mean(f_i))`
3. For new image, compute similarity: `sim(f, c_k) = f · c_k`
4. Apply temperature and softmax: `p_k = softmax(T × sim)`

## Configuration Options

### Monitoring Parameters

Edit `main.py` to adjust monitoring behavior:

```python
check_interval = 3  # seconds between checks (default: 3)
temperature = 100.0  # confidence scaling in predict() (default: 100.0)
```

### Classification Customization

To adapt for different use cases, modify the class mappings in `main.py`:

```python
CLASSES = ["open", "closed"]  # Your class names
SUPPORT_FOLDERS = ["fewshot/pos", "fewshot/neg"]  # Corresponding folders
```

## Gmail Setup

For Gmail users, you need to use an App Password:
1. Enable 2-factor authentication on your Google account
2. Generate an App Password at https://myaccount.google.com/apppasswords
3. Use the App Password (not your regular password) in `local.env.sh`

## SMTP Configuration

The default SMTP settings are configured for Gmail. For other email providers:

- **Outlook/Hotmail**: `smtp.office365.com`, port `587`
- **Yahoo**: `smtp.mail.yahoo.com`, port `587`
- **Custom**: Specify your SMTP server and port

```python
send_email_with_photo(
    sender_email="your_email@outlook.com",
    sender_password="your_password",
    recipient_email="recipient@example.com",
    subject="Alert",
    body="State changed!",
    photo_path="photo.jpg",
    smtp_server="smtp.office365.com",
    smtp_port=587
)
```

## Project Structure

```
CamClassifier/
├── main.py                 # Main monitoring script
├── predict.py              # Classification logic with prototypical networks
├── capture_photo.py        # Camera capture functionality
├── send_email.py           # Email sending utilities
├── capture_and_send.py     # Combined capture + send
├── requirements.txt        # Python dependencies
├── local.env.sh           # Environment variables (not in git)
├── fewshot/               # Few-shot example images
│   ├── pos/               # Positive class examples (e.g., "open")
│   └── neg/               # Negative class examples (e.g., "closed")
└── pics/                  # Captured photos directory
```

## Troubleshooting

### Camera Access
- **macOS**: Grant camera permissions in System Preferences → Security & Privacy → Camera
- **Error**: If capture fails, check that no other app is using the camera

### Email Issues
- Verify credentials in `local.env.sh`
- For Gmail, ensure you're using an App Password, not your regular password
- Check SMTP server and port settings

### Model Performance
- **Low confidence**: Add more example images to `fewshot/pos` and `fewshot/neg`
- **Wrong predictions**: Ensure examples are representative of the states you want to detect
- **Slow inference**: On Apple Silicon, verify MPS is being used (check console output)

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! This project can be adapted for various monitoring tasks:
- Pet detection (home/away)
- Package delivery monitoring
- Parking spot availability
- Plant health monitoring
- Any visual state change detection