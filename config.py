"""
Configuration management for Traffic Signal Violation Detection System.

This module centralizes all configuration parameters, file paths, and model settings
to eliminate hardcoded values and improve portability across different systems.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory (parent of this config file)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Model weights
WEIGHTS_DIR = PROJECT_ROOT / "weights"
YOLOV3_WEIGHTS_PATH = WEIGHTS_DIR / "yolov3.weights"
WEIGHTS_PATH = YOLOV3_WEIGHTS_PATH  # Alias for backward compatibility

# Resources directories
RESOURCES_DIR = PROJECT_ROOT / "Resources"
INPUT_DIR = RESOURCES_DIR / "input"
OUTPUT_DIR = RESOURCES_DIR / "output"

# Image directories
IMAGES_DIR = PROJECT_ROOT / "Images"
PREVIEW_IMAGE_PATH = IMAGES_DIR / "preview.jpg"
COPY_IMAGE_PATH = IMAGES_DIR / "copy.jpg"
HOME_IMAGE_PATH = IMAGES_DIR / "home.jpg"

# Detected violations
DETECTED_IMAGES_DIR = PROJECT_ROOT / "Detected Images"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================================================
# YOLO MODEL PARAMETERS
# ============================================================================

# Network input dimensions
NET_H = 416
NET_W = 416

# Detection thresholds
OBJ_THRESH = 0.5  # Objectness threshold for detection confidence
NMS_THRESH = 0.45  # Non-maximum suppression threshold

# YOLOv3 anchor boxes for different scales
ANCHORS = [
    [116, 90, 156, 198, 373, 326],  # Large objects
    [30, 61, 62, 45, 59, 119],       # Medium objects
    [10, 13, 16, 30, 33, 23]         # Small objects
]

# COCO dataset class labels (80 classes)
LABELS = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Vehicle class indices for filtering (car, motorbike, bus, truck)
VEHICLE_CLASSES = [2, 3, 5, 7]

# ============================================================================
# GUI PARAMETERS
# ============================================================================

# Main window dimensions
GUI_WIDTH = 535
GUI_HEIGHT = 380

# Canvas dimensions
CANVAS_WIDTH = 1366
CANVAS_HEIGHT = 768

# ============================================================================
# VIDEO PROCESSING PARAMETERS
# ============================================================================

# Output video settings
OUTPUT_VIDEO_NAME = "output.mp4"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / OUTPUT_VIDEO_NAME

# Frame processing
SKIP_FRAMES = 1  # Process every Nth frame (1 = process all frames)

# Violation detection
VIOLATION_IMAGE_PREFIX = "violation_"

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "traffic_violation.log"

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# GPU configuration
USE_GPU = True  # Enable GPU acceleration if available
GPU_MEMORY_GROWTH = True  # Allow dynamic GPU memory allocation

# Batch processing
BATCH_SIZE = 1  # Number of frames to process simultaneously

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories():
    """
    Create all necessary directories if they don't exist.

    This should be called at application startup to ensure the directory
    structure is properly initialized.
    """
    directories = [
        WEIGHTS_DIR,
        INPUT_DIR,
        OUTPUT_DIR,
        IMAGES_DIR,
        DETECTED_IMAGES_DIR,
        LOGS_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_output_video_path(custom_name=None):
    """
    Get the output video path, optionally with a custom filename.

    Args:
        custom_name (str, optional): Custom filename for output video

    Returns:
        Path: Full path to output video file
    """
    if custom_name:
        return OUTPUT_DIR / custom_name
    return OUTPUT_VIDEO_PATH


def get_violation_image_path(frame_number):
    """
    Get the path for saving a violation snapshot image.

    Args:
        frame_number (int): Frame number where violation occurred

    Returns:
        Path: Full path to save violation image
    """
    filename = f"{VIOLATION_IMAGE_PREFIX}{frame_number}.jpg"
    return DETECTED_IMAGES_DIR / filename


# ============================================================================
# INITIALIZATION
# ============================================================================

# Ensure all directories exist when config is imported
ensure_directories()


# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def configure_gpu():
    """
    Configure GPU settings for TensorFlow.

    Automatically detects available GPUs and configures:
    - Memory growth to prevent OOM errors
    - Optional memory limit
    - Multi-GPU selection

    Returns:
        list: Names of available GPU devices

    Example:
        >>> gpus = configure_gpu()
        >>> print(f"Using {len(gpus)} GPU(s)")
    """
    import logging

    try:
        import tensorflow as tf
    except ImportError:
        logging.warning("TensorFlow not installed. GPU configuration skipped.")
        return []

    # Reduce TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    try:
        # List all physical GPUs
        gpus = tf.config.list_physical_devices('GPU')

        if not gpus:
            logging.info("No GPU detected. Running on CPU.")
            return []

        if not USE_GPU:
            # Force CPU mode
            logging.info("GPU available but USE_GPU=False. Running on CPU.")
            tf.config.set_visible_devices([], 'GPU')
            return []

        # Configure each GPU
        gpu_names = []

        for gpu in gpus:
            gpu_name = gpu.name
            gpu_names.append(gpu_name)

            # Enable memory growth (allocate memory as needed)
            # This prevents TensorFlow from allocating all GPU memory at once
            if GPU_MEMORY_GROWTH:
                tf.config.experimental.set_memory_growth(gpu, True)

            logging.info(f"✅ GPU configured: {gpu_name}")

        # Log GPU details
        logging.info(f"TensorFlow GPU support: {tf.test.is_built_with_cuda()}")
        logging.info(f"GPUs available: {len(gpus)}")

        for i, gpu in enumerate(gpus):
            logging.info(f"  GPU {i}: {gpu.name}")

        return gpu_names

    except RuntimeError as e:
        # Memory growth must be set before GPUs are initialized
        logging.error(f"GPU configuration error: {e}")
        logging.info("Falling back to CPU mode")
        return []

    except Exception as e:
        logging.error(f"Unexpected GPU configuration error: {e}")
        logging.info("Falling back to CPU mode")
        return []


def get_device_info():
    """
    Get detailed information about computation device.

    Returns:
        dict: Device information
            {
                'device_type': str ('GPU' or 'CPU'),
                'device_count': int,
                'device_names': list[str],
                'tensorflow_version': str,
                'cuda_available': bool
            }

    Example:
        >>> info = get_device_info()
        >>> print(f"Running on {info['device_type']}")
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')

        info = {
            'device_type': 'GPU' if (gpus and USE_GPU) else 'CPU',
            'device_count': len(gpus) if gpus else 0,
            'device_names': [gpu.name for gpu in gpus] if gpus else [],
            'tensorflow_version': tf.__version__,
            'cuda_available': tf.test.is_built_with_cuda()
        }

        return info

    except ImportError:
        return {
            'device_type': 'CPU',
            'device_count': 0,
            'device_names': [],
            'tensorflow_version': 'not installed',
            'cuda_available': False
        }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """
    Validate configuration settings and check for required files.

    Returns:
        tuple: (is_valid: bool, errors: list)
    """
    errors = []

    # Check if weights file exists
    if not YOLOV3_WEIGHTS_PATH.exists():
        errors.append(
            f"YOLOv3 weights file not found at {YOLOV3_WEIGHTS_PATH}. "
            f"Download from: https://pjreddie.com/media/files/yolov3.weights"
        )

    # Check if home image exists
    if not HOME_IMAGE_PATH.exists():
        errors.append(
            f"Home image not found at {HOME_IMAGE_PATH}. "
            f"This is required for the GUI startup screen."
        )

    # Validate thresholds
    if not 0 < OBJ_THRESH < 1:
        errors.append(f"OBJ_THRESH must be between 0 and 1, got {OBJ_THRESH}")

    if not 0 < NMS_THRESH < 1:
        errors.append(f"NMS_THRESH must be between 0 and 1, got {NMS_THRESH}")

    # Validate network dimensions
    if NET_H <= 0 or NET_W <= 0:
        errors.append(f"Network dimensions must be positive, got {NET_H}x{NET_W}")

    is_valid = len(errors) == 0
    return is_valid, errors


if __name__ == "__main__":
    # When run directly, perform validation and print configuration
    print("=" * 70)
    print("Traffic Signal Violation Detection System - Configuration")
    print("=" * 70)
    print()
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Weights Path: {YOLOV3_WEIGHTS_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Detected Images: {DETECTED_IMAGES_DIR}")
    print()
    print("Model Parameters:")
    print(f"  Network Size: {NET_W}x{NET_H}")
    print(f"  Object Threshold: {OBJ_THRESH}")
    print(f"  NMS Threshold: {NMS_THRESH}")
    print()

    # Validate configuration
    is_valid, errors = validate_config()

    if is_valid:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration errors found:")
        for error in errors:
            print(f"  - {error}")

    print()
    print("=" * 70)
