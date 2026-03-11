#!/usr/bin/env python3
"""
List all available cameras: OpenCV indices and system camera names (macOS).
Run from project root with:  source local.env.sh && python list_cameras.py
"""
import subprocess
import sys

def _log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def list_opencv_cameras():
    """Try indices 0..9 and report which open and their resolution."""
    import cv2
    _log("OpenCV camera indices (cv2.VideoCapture):")
    _log("-" * 50)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            _log(f"  Index {i}: not available")
            try:
                cap.release()
            except Exception:
                pass
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            _log(f"  Index {i}: OK  (resolution: {w} x {h})")
        else:
            _log(f"  Index {i}: opened but read failed")
        cap.release()
    _log()


def list_system_cameras_macos():
    """Print macOS camera names from system_profiler."""
    _log("System cameras (macOS):")
    _log("-" * 50)
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            _log("  (system_profiler failed)")
            return
        # Parse simple name lines like "    FaceTime高清相机:"
        for line in out.stdout.splitlines():
            line = line.strip()
            if line and not line.startswith("Model ID:") and not line.startswith("Unique ID:"):
                if line.endswith(":"):
                    name = line[:-1].strip()
                    if name and name != "Camera":
                        _log(f"  - {name}")
    except FileNotFoundError:
        _log("  (system_profiler not found)")
    except Exception as e:
        _log(f"  ({e})")
    _log()


if __name__ == "__main__":
    list_system_cameras_macos()
    list_opencv_cameras()
    _log("OpenCV does not expose camera names; index order may not match the system list above.")
    _log("Set CAMERA_INDEX in local.env.sh to 0, 1, or 2 and run monitoring to see which is your webcam.")
