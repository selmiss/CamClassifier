"""
Resolve camera index by device name so the correct webcam is used even when indices change.

On macOS, OpenCV (AVFoundation) enumerates cameras sorted by Unique ID, which is
*not* the same as system_profiler's display order. We parse names + Unique IDs,
sort by Unique ID to get OpenCV order, then find the webcam by name.
"""
import os
import re
import subprocess
import sys


# Name patterns that identify the web camera (case-insensitive).
WEBCAM_NAME_PATTERNS = [
    r"web\s*cam",      # "Web Cam", "WebCam", "Web camera"
    r"wed\s*cam",      # "Wed Camera" (typo / localized)
    r"^webcam\b",
    r"external\s*cam",
]


def get_system_cameras_with_ids_macos():
    """
    Return list of (name, unique_id) for each camera, in the order listed by
    system_profiler. Caller must sort by unique_id to get OpenCV index order.
    Returns [] on non-macOS or on error.
    """
    if sys.platform != "darwin":
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return []
        cameras = []  # (name, unique_id)
        current_name = None
        for line in out.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Unique ID:"):
                unique_id = stripped.split("Unique ID:", 1)[1].strip()
                if current_name is not None:
                    cameras.append((current_name, unique_id))
                    current_name = None
                continue
            if stripped.startswith("Model ID:"):
                continue
            if stripped.endswith(":") and ":" in stripped:
                name = stripped[:-1].strip()
                if name and name != "Camera":
                    current_name = name
        return cameras
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def _name_matches_webcam(name):
    if not name:
        return False
    lower = name.lower()
    for pat in WEBCAM_NAME_PATTERNS:
        if re.search(pat, lower, re.IGNORECASE):
            return True
    return False


def get_webcam_index(log_fn=None):
    """
    Detect the OpenCV camera index for the web camera by matching its system name.
    On macOS we sort cameras by Unique ID (to match OpenCV/AVFoundation order),
    then find the index of the camera whose name matches the webcam pattern.

    Returns:
        int: Index to use with cv2.VideoCapture(index).
        Fallback: CAMERA_INDEX env (if set), then 0.
    """
    log = log_fn if callable(log_fn) else lambda _: None
    cameras = get_system_cameras_with_ids_macos()
    if not cameras:
        idx = int(os.environ.get("CAMERA_INDEX", "0"))
        log(f"Could not list system cameras; using CAMERA_INDEX/env fallback: {idx}")
        return idx
    # OpenCV on macOS orders devices by Unique ID (ascending). Sort to get same order.
    cameras_sorted = sorted(cameras, key=lambda x: (x[1], x[0]))
    for i, (name, _) in enumerate(cameras_sorted):
        if _name_matches_webcam(name):
            log(f"Using web camera at index {i}: '{name}' (OpenCV order by Unique ID)")
            return i
    idx = int(os.environ.get("CAMERA_INDEX", "0"))
    names_only = [n for n, _ in cameras_sorted]
    log(f"No web camera name matched (saw: {names_only}); using fallback index {idx}")
    return idx
