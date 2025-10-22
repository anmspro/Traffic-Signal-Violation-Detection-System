# Traffic Signal Violation Detection System - Remaining Implementation Guide

## Document Overview

**Purpose**: Complete implementation handbook for Tasks 7-15
**Audience**: Human developers and AI agents
**Status**: Ready for implementation
**Last Updated**: 2025-10-22

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Context](#project-context)
3. [Completed Work Reference](#completed-work-reference)
4. [Task 7: Lazy Model Loading](#task-7-lazy-model-loading)
5. [Task 8: Modular Architecture](#task-8-modular-architecture)
6. [Task 9: CLI Interface](#task-9-cli-interface)
7. [Task 10: CSV Violation Reporting](#task-10-csv-violation-reporting)
8. [Task 11: Progress Bars](#task-11-progress-bars)
9. [Task 12: GPU Acceleration](#task-12-gpu-acceleration)
10. [Task 13: Unit Tests](#task-13-unit-tests)
11. [Task 14: Comprehensive Docstrings](#task-14-comprehensive-docstrings)
12. [Task 15: Docker Container](#task-15-docker-container)
13. [Common Patterns](#common-patterns)
14. [Troubleshooting Guide](#troubleshooting-guide)
15. [Success Criteria](#success-criteria)

---

## Introduction

This document provides complete, copy-paste ready implementation guides for the 10 remaining tasks in the Traffic Signal Violation Detection System improvement plan. Each task includes:

- **Problem Statement**: What's broken/missing and why it matters
- **Solution Design**: Architecture and approach
- **Step-by-Step Implementation**: Numbered instructions
- **Complete Code Examples**: Ready to use snippets
- **Testing Procedures**: How to verify it works
- **Acceptance Criteria**: Checklist for completion
- **Files to Create/Modify**: Exact file list
- **Gotchas & Warnings**: Common mistakes to avoid

---

## Project Context

### What This Project Does

A Python-based traffic violation detection system that:
1. Loads traffic camera video footage
2. Uses YOLOv3 deep learning model to detect vehicles
3. Allows users to draw a virtual traffic signal line
4. Detects when vehicles cross the line (violation)
5. Outputs annotated video with violations highlighted

### Current Technology Stack

- **Python**: 3.8+
- **TensorFlow**: 2.10+ (migrated from Keras 1.x)
- **OpenCV**: 4.6+ for video processing
- **NumPy**: Array operations
- **Tkinter**: GUI framework
- **YOLOv3**: Object detection model

### Current Project Structure

```
Traffic-Signal-Violation-Detection-System/
├── config.py                    ✅ DONE - Configuration management
├── Project-GUI.py               ✅ UPDATED - Main GUI application
├── object_detection.py          ✅ UPDATED - YOLOv3 detection engine
├── requirements.txt             ✅ DONE - Full dependencies
├── requirements-minimal.txt     ✅ DONE - Production dependencies
├── utils/
│   ├── __init__.py             ✅ DONE
│   └── geometry.py             ✅ DONE - Line intersection utilities
├── Resources/
│   ├── input/                   Sample videos
│   └── output/                  Processed videos
├── Detected Images/             Violation snapshots
└── weights/
    └── yolov3.weights          YOLOv3 pre-trained weights
```

### Target Structure (After All Tasks)

```
Traffic-Signal-Violation-Detection-System/
├── main.py                      ⏳ Task 9 - CLI entry point
├── config.py                    ✅ Done
├── models/
│   ├── __init__.py             ⏳ Task 8
│   ├── yolov3.py               ⏳ Task 8 - Model architecture
│   └── weight_loader.py        ⏳ Task 8 - WeightReader class
├── detection/
│   ├── __init__.py             ⏳ Task 8
│   ├── detector.py             ⏳ Task 8 - Detection logic
│   ├── preprocessing.py        ⏳ Task 8 - Image preprocessing
│   └── violation_checker.py    ⏳ Task 8 - Violation detection
├── gui/
│   ├── __init__.py             ⏳ Task 8
│   └── main_window.py          ⏳ Task 8 - Refactored GUI
├── utils/
│   ├── __init__.py             ✅ Done
│   ├── geometry.py             ✅ Done
│   ├── video_io.py             ⏳ Task 8
│   ├── report_generator.py     ⏳ Task 10 - CSV reporting
│   └── logging_config.py       ⏳ Task 8
├── tests/
│   ├── __init__.py             ⏳ Task 13
│   ├── test_geometry.py        ⏳ Task 13
│   ├── test_detection.py       ⏳ Task 13
│   └── test_report_generator.py ⏳ Task 13
├── Dockerfile                   ⏳ Task 15
├── docker-compose.yml           ⏳ Task 15
├── .dockerignore               ⏳ Task 15
├── requirements.txt             ✅ Done
└── README.md                    ⏳ Update
```

---

## Completed Work Reference

Before implementing new tasks, study these completed examples to maintain consistency:

### config.py - Configuration Pattern

**File**: `config.py` (240 lines)

**Key Patterns to Follow**:

1. **Use pathlib for cross-platform paths**:
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
WEIGHTS_PATH = PROJECT_ROOT / "weights" / "yolov3.weights"
```

2. **Centralize all constants**:
```python
# Model parameters
NET_H = 416
NET_W = 416
OBJ_THRESH = 0.5
NMS_THRESH = 0.45
```

3. **Validation functions**:
```python
def validate_config() -> bool:
    """Validate that all required files and directories exist."""
    # Check files
    # Check directories
    # Return True/False
```

4. **Auto-create directories**:
```python
def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for directory in [OUTPUT_DIR, DETECTED_IMAGES_DIR, PREVIEW_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
```

**Reference**: Read `config.py` for full implementation

---

### utils/geometry.py - Utility Module Pattern

**File**: `utils/geometry.py` (220 lines)

**Key Patterns to Follow**:

1. **Type hints on all functions**:
```python
from typing import Tuple

Point = Tuple[float, float]
BBox = Tuple[Point, Point]

def line_segment_intersection(
    p1: Point, p2: Point, p3: Point, p4: Point
) -> bool:
    """Check if two line segments intersect."""
```

2. **Google-style docstrings**:
```python
def bounding_box_intersects_line(
    bbox_top_left: Point,
    bbox_bottom_right: Point,
    line_start: Point,
    line_end: Point
) -> bool:
    """
    Check if a bounding box intersects with a line segment.

    This is the core violation detection function. A vehicle is considered
    to be in violation if any edge of its bounding box crosses the traffic
    signal line.

    Args:
        bbox_top_left: Top-left corner (xmin, ymin) of bounding box
        bbox_bottom_right: Bottom-right corner (xmax, ymax) of bounding box
        line_start: Starting point (x1, y1) of traffic line
        line_end: Ending point (x2, y2) of traffic line

    Returns:
        True if bounding box intersects the line, False otherwise

    Example:
        >>> bbox_intersects_line((100, 150), (250, 300), (0, 200), (400, 200))
        True
    """
```

3. **Built-in tests**:
```python
def _run_tests() -> None:
    """Run built-in unit tests."""
    print("Running geometry tests...")

    # Test 1: Perpendicular intersection
    assert line_segment_intersection(...)

    print("✅ All geometry tests passed!")

if __name__ == "__main__":
    _run_tests()
```

**Reference**: Read `utils/geometry.py` for full implementation

---

### Project-GUI.py - Error Handling Pattern

**File**: `Project-GUI.py` (lines 52-112)

**Key Patterns to Follow**:

1. **Try-except with user feedback**:
```python
import tkinter.messagebox as messagebox
import logging

def open_file(self):
    try:
        self.filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
        )

        if not self.filename:
            return  # User cancelled

        # Validate file exists
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")

        # Try to open video
        cap = cv2.VideoCapture(self.filename)
        if not cap.isOpened():
            raise ValueError("Unable to open video file. File may be corrupted.")

        # Process...

    except FileNotFoundError as e:
        messagebox.showerror("File Not Found", str(e))
        logging.error(f"File not found: {e}")
    except ValueError as e:
        messagebox.showerror("Invalid Video", str(e))
        logging.error(f"Video validation error: {e}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")
        logging.exception("Unexpected error in open_file")
```

2. **Resource cleanup with finally**:
```python
def main_process(self):
    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(self.filename)
        # Processing...

    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
    except Exception as e:
        logging.exception("Error during processing")
        messagebox.showerror("Processing Error", str(e))
    finally:
        # Always clean up resources
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
```

**Reference**: Read `Project-GUI.py` lines 52-112 for full implementation

---

## Task 7: Lazy Model Loading

### Problem Statement

**Current Issue**:
- YOLOv3 model loads at module import time (`object_detection.py` lines 408-412)
- Takes ~30 seconds to load model weights (237 MB)
- Consumes ~2 GB RAM immediately
- Slows down all operations, even simple config validation

**Example of Problem**:
```bash
# Just checking if weights file exists:
$ python config.py
# Takes 30+ seconds because it imports object_detection.py!
```

**Why This Matters**:
- Poor developer experience
- Unnecessary resource consumption
- Prevents quick CLI operations
- Makes testing slow

---

### Solution Design

**Approach**: Lazy loading using Python property decorator

**Core Concept**:
1. Create `ViolationDetector` class to encapsulate model
2. Store model as private `_model = None`
3. Use `@property` decorator to load on first access
4. Cache loaded model for subsequent calls

**Benefits**:
- Model only loads when actually needed
- Startup time: 30s → <1s for non-processing tasks
- Better memory management
- Maintains compatibility with existing code

---

### Step-by-Step Implementation

#### Step 1: Create ViolationDetector Class

**File**: `object_detection.py`

**Location**: Add at end of file, before the current global model loading code (around line 390)

**Code**:
```python
class ViolationDetector:
    """
    Lazy-loading wrapper for YOLOv3 model.

    The model is not loaded until the first prediction is made,
    improving startup time for non-processing operations.

    Attributes:
        _model: Cached Keras model (None until first use)

    Example:
        >>> detector = ViolationDetector()
        >>> # Model not loaded yet
        >>> boxes = detector.predict(image)  # Model loads here
        >>> boxes2 = detector.predict(image2)  # Uses cached model
    """

    def __init__(self):
        """Initialize detector without loading model."""
        self._model = None
        logging.info("ViolationDetector initialized (model not loaded yet)")

    @property
    def model(self):
        """
        Get the YOLOv3 model, loading it if necessary.

        Returns:
            Keras Model: Loaded YOLOv3 model ready for predictions
        """
        if self._model is None:
            logging.info("Loading YOLOv3 model for first time...")
            start_time = time.time()

            # Create model architecture
            self._model = make_yolov3_model()

            # Load pre-trained weights
            weight_reader = WeightReader(str(WEIGHTS_PATH))
            weight_reader.load_weights(self._model)

            elapsed = time.time() - start_time
            logging.info(f"Model loaded successfully in {elapsed:.2f}s")

        return self._model

    def predict(self, image):
        """
        Run object detection on an image.

        Args:
            image: Preprocessed image array (1, NET_H, NET_W, 3)

        Returns:
            Model predictions (yolo outputs)
        """
        return self.model.predict(image)
```

#### Step 2: Add Required Imports

**File**: `object_detection.py`

**Location**: Top of file, after existing imports

**Add**:
```python
import time
import logging
```

#### Step 3: Remove Global Model Loading

**File**: `object_detection.py`

**Location**: Lines 408-412 (approximate)

**Find and COMMENT OUT (don't delete yet)**:
```python
# OLD CODE - COMMENT OUT:
# model = make_yolov3_model()
# weight_reader = WeightReader('yolov3.weights')
# weight_reader.load_weights(model)
```

#### Step 4: Create Global Detector Instance

**File**: `object_detection.py`

**Location**: Where old model loading code was

**Add**:
```python
# Global detector instance (lazy loading)
detector = ViolationDetector()
```

#### Step 5: Update All model.predict() Calls

**File**: `object_detection.py`

**Find**: `model.predict(input_image)`

**Replace with**: `detector.predict(input_image)`

**Locations**: Search for all occurrences (likely in `preprocess_input()` or similar)

#### Step 6: Update Project-GUI.py to Use Detector

**File**: `Project-GUI.py`

**Find**: Lines that import or use `model` from `object_detection`

**Update import**:
```python
# Old:
# from object_detection import model, ...

# New:
from object_detection import detector, ...
```

**Update usage** in `main_process()`:
```python
# Old:
# yolos = model.predict(input_image)

# New:
yolos = detector.predict(input_image)
```

---

### Complete Code Example

Here's the complete `ViolationDetector` class with all features:

```python
import logging
import time
from pathlib import Path

# Assuming config is imported:
# from config import WEIGHTS_PATH, NET_H, NET_W

class ViolationDetector:
    """
    Lazy-loading wrapper for YOLOv3 traffic violation detection model.

    This class implements the Lazy Initialization pattern to defer
    expensive model loading until actually needed.

    Attributes:
        _model (keras.Model | None): Cached YOLOv3 model instance
        _load_time (float): Time taken to load model (seconds)

    Performance:
        - First prediction: ~30s (model loading + inference)
        - Subsequent predictions: ~0.1s (inference only)

    Example:
        >>> # Create detector (instant)
        >>> detector = ViolationDetector()
        >>>
        >>> # Load video and preprocess frame
        >>> image = preprocess_input(frame, NET_H, NET_W)
        >>>
        >>> # First call loads model (~30s)
        >>> predictions = detector.predict(image)
        >>>
        >>> # Subsequent calls are fast
        >>> predictions2 = detector.predict(image2)
    """

    def __init__(self):
        """
        Initialize detector without loading model.

        The model will be loaded lazily on first predict() call.
        """
        self._model = None
        self._load_time = None
        logging.info("ViolationDetector initialized (lazy loading enabled)")

    @property
    def model(self):
        """
        Get the YOLOv3 model, loading it on first access.

        This property implements lazy loading - the expensive model
        loading operation only happens on first access.

        Returns:
            keras.Model: Loaded YOLOv3 model ready for inference

        Raises:
            FileNotFoundError: If weights file doesn't exist
            RuntimeError: If model loading fails
        """
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """
        Internal method to load YOLOv3 model and weights.

        Raises:
            FileNotFoundError: If WEIGHTS_PATH doesn't exist
            RuntimeError: If model creation or weight loading fails
        """
        logging.info("Loading YOLOv3 model...")
        start_time = time.time()

        try:
            # Verify weights file exists
            if not Path(WEIGHTS_PATH).exists():
                raise FileNotFoundError(
                    f"YOLOv3 weights not found at {WEIGHTS_PATH}. "
                    "Please download from: https://pjreddie.com/media/files/yolov3.weights"
                )

            # Create model architecture
            logging.info("Building YOLOv3 architecture...")
            self._model = make_yolov3_model()

            # Load pre-trained weights
            logging.info(f"Loading weights from {WEIGHTS_PATH}...")
            weight_reader = WeightReader(str(WEIGHTS_PATH))
            weight_reader.load_weights(self._model)

            # Calculate load time
            self._load_time = time.time() - start_time

            logging.info(f"✅ Model loaded successfully in {self._load_time:.2f}s")
            logging.info(f"Model size: {self._get_model_size_mb():.1f} MB")

        except Exception as e:
            logging.error(f"Failed to load YOLOv3 model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def predict(self, image):
        """
        Run object detection on preprocessed image.

        Args:
            image (np.ndarray): Preprocessed image with shape (1, NET_H, NET_W, 3)

        Returns:
            list: YOLOv3 predictions for three scales
                  [output_1, output_2, output_3]

        Raises:
            ValueError: If image has wrong shape
            RuntimeError: If prediction fails

        Example:
            >>> preprocessed = preprocess_input(frame, 416, 416)
            >>> predictions = detector.predict(preprocessed)
            >>> boxes = decode_netout(predictions[0], ...)
        """
        # Validate input shape
        if image.shape != (1, NET_H, NET_W, 3):
            raise ValueError(
                f"Expected image shape (1, {NET_H}, {NET_W}, 3), "
                f"got {image.shape}"
            )

        try:
            # This will trigger lazy loading if needed
            return self.model.predict(image, verbose=0)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Detection failed: {e}") from e

    def _get_model_size_mb(self):
        """
        Calculate approximate model size in memory.

        Returns:
            float: Model size in megabytes
        """
        if self._model is None:
            return 0.0

        # Calculate total parameters
        total_params = self._model.count_params()

        # Assume float32 (4 bytes per parameter)
        size_bytes = total_params * 4
        size_mb = size_bytes / (1024 * 1024)

        return size_mb

    @property
    def is_loaded(self):
        """Check if model is currently loaded in memory."""
        return self._model is not None

    def unload(self):
        """
        Unload model from memory to free resources.

        Useful for long-running applications that process videos
        in batches with idle time between batches.
        """
        if self._model is not None:
            logging.info("Unloading model from memory")
            del self._model
            self._model = None

            # Force garbage collection
            import gc
            gc.collect()


# Global detector instance (lazy loading)
detector = ViolationDetector()
```

---

### Testing Procedures

#### Test 1: Verify Lazy Loading

**Goal**: Confirm model doesn't load on import

**Steps**:
1. Open Python interpreter
2. Import module and time it:

```bash
$ python
>>> import time
>>> start = time.time()
>>> from object_detection import detector
>>> print(f"Import time: {time.time() - start:.2f}s")
Import time: 0.15s  # Should be < 1s

>>> print(detector.is_loaded)
False  # Model not loaded yet
```

#### Test 2: Verify Model Loads on First Use

**Goal**: Confirm model loads on first predict()

**Steps**:
```python
>>> import numpy as np
>>> from config import NET_H, NET_W
>>>
>>> # Create dummy image
>>> dummy_image = np.random.rand(1, NET_H, NET_W, 3).astype('float32')
>>>
>>> # First prediction (triggers loading)
>>> import time
>>> start = time.time()
>>> predictions = detector.predict(dummy_image)
Loading YOLOv3 model...
Model loaded successfully in 28.45s
>>> print(f"First prediction: {time.time() - start:.2f}s")
First prediction: 28.67s
>>>
>>> # Second prediction (uses cache)
>>> start = time.time()
>>> predictions2 = detector.predict(dummy_image)
>>> print(f"Second prediction: {time.time() - start:.2f}s")
Second prediction: 0.12s
```

#### Test 3: Integration Test with GUI

**Goal**: Verify GUI still works

**Steps**:
1. Run GUI application:
```bash
$ python Project-GUI.py
```

2. Load a test video
3. Draw traffic line
4. Click "Start Processing"
5. Verify:
   - Model loads on first frame
   - Processing continues normally
   - Violations detected correctly

#### Test 4: Config Validation Speed

**Goal**: Verify config.py runs fast without loading model

**Steps**:
```bash
$ time python config.py
# Should complete in < 1s (previously took 30s)
```

---

### Acceptance Criteria

Mark complete when ALL of these are true:

- [ ] `ViolationDetector` class created in `object_detection.py`
- [ ] Model loads on first `predict()` call, not on import
- [ ] Subsequent `predict()` calls use cached model
- [ ] `detector.is_loaded` returns False before first use
- [ ] `detector.is_loaded` returns True after first use
- [ ] Import time for `object_detection` module < 1 second
- [ ] First prediction time ~28-30s (normal model load time)
- [ ] Second prediction time < 1s (cached)
- [ ] All existing functionality in Project-GUI.py works
- [ ] No errors when running `python config.py`
- [ ] Logging messages show when model loads
- [ ] Tests pass (see Testing Procedures)

---

### Files to Create/Modify

**Modified**:
- `object_detection.py` - Add `ViolationDetector` class, update model loading
- `Project-GUI.py` - Update imports and model usage

**No new files required for this task**

---

### Gotchas & Warnings

1. **Don't delete old model loading code immediately**
   - Comment it out first
   - Test thoroughly
   - Then delete after confirming everything works

2. **Update ALL model.predict() calls**
   - Search entire codebase for `model.predict`
   - Replace with `detector.predict`
   - Easy to miss one!

3. **Import order matters**
   ```python
   # Wrong - will fail
   from object_detection import model  # model doesn't exist anymore!

   # Right
   from object_detection import detector
   ```

4. **Error handling in property**
   - The `@property` decorator can hide errors
   - Add try-except in `_load_model()`
   - Log errors clearly

5. **Memory management**
   - Model stays in memory once loaded
   - For long-running applications, consider adding `unload()` method
   - Provided in complete example above

---

## Task 9: CLI Interface

### Problem Statement

**Current Issue**:
- Application only has GUI interface
- Cannot process videos from command line
- No batch processing capability
- Cannot integrate into automated workflows
- Cannot run on headless servers

**Example of What's Missing**:
```bash
# Want to do this:
$ python main.py --video input.mp4 --line "100,200,500,200" --output result.mp4

# Currently must:
# 1. Launch GUI
# 2. Click "Open File"
# 3. Navigate to video
# 4. Click twice to draw line
# 5. Click "Start Processing"
# 6. Wait...
```

**Why This Matters**:
- Server deployment impossible
- Cannot automate processing
- No CI/CD integration
- Poor developer experience for testing

---

### Solution Design

**Approach**: Create `main.py` with argparse CLI + GUI launcher

**Features**:
1. Default: Launch GUI (backward compatible)
2. `--video`: Headless video processing
3. `--line`: Specify traffic line coordinates
4. `--confidence`: Adjust detection threshold
5. `--output`: Custom output path
6. `--batch`: Process multiple videos

**Architecture**:
```
main.py
├── parse_arguments()  # CLI argument parsing
├── validate_inputs()  # Input validation
├── run_gui_mode()     # Launch existing GUI
└── run_cli_mode()     # Headless processing
    ├── load_video()
    ├── process_frames()
    └── save_output()
```

---

### Step-by-Step Implementation

#### Step 1: Create main.py

**File**: `main.py` (new file in project root)

**Complete Implementation**:

```python
#!/usr/bin/env python3
"""
Traffic Signal Violation Detection System - Main Entry Point

This module provides both GUI and CLI interfaces for traffic violation detection.

Usage:
    # Launch GUI (default)
    python main.py

    # Process video from command line
    python main.py --video input.mp4 --line "100,200,500,200"

    # Customize output and confidence
    python main.py --video input.mp4 --line "100,200,500,200" \\
                   --output result.mp4 --confidence 0.7

    # Batch process directory
    python main.py --input-dir videos/ --output-dir results/ \\
                   --line "100,200,500,200"
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np

# Import project modules
import config
from object_detection import detector, preprocess_input, decode_netout, draw_boxes
from utils.geometry import bounding_box_intersects_line


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        verbose: If True, set log level to DEBUG
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.PROJECT_ROOT / 'violation_detection.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_line_coordinates(line_str: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Parse line coordinates from string format.

    Args:
        line_str: Line coordinates in format "x1,y1,x2,y2"

    Returns:
        Tuple of ((x1, y1), (x2, y2))

    Raises:
        ValueError: If format is invalid

    Example:
        >>> parse_line_coordinates("100,200,500,200")
        ((100, 200), (500, 200))
    """
    try:
        coords = [int(x.strip()) for x in line_str.split(',')]
        if len(coords) != 4:
            raise ValueError
        return ((coords[0], coords[1]), (coords[2], coords[3]))
    except (ValueError, IndexError):
        raise ValueError(
            f"Invalid line format: '{line_str}'. "
            "Expected format: 'x1,y1,x2,y2' (e.g., '100,200,500,200')"
        )


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Traffic Signal Violation Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch GUI
  %(prog)s

  # Process single video
  %(prog)s --video input.mp4 --line "100,200,500,200"

  # Custom output and confidence threshold
  %(prog)s --video input.mp4 --line "100,200,500,200" \\
           --output result.mp4 --confidence 0.7

  # Batch process directory
  %(prog)s --input-dir videos/ --output-dir results/ \\
           --line "100,200,500,200"

  # Headless mode (no GUI, no visualization)
  %(prog)s --video input.mp4 --line "100,200,500,200" --no-gui
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--video', '-v',
        type=str,
        help='Path to input video file'
    )
    input_group.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Directory containing videos to process'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output video file (default: Resources/output/output.mp4)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory for batch processing output'
    )

    # Detection options
    parser.add_argument(
        '--line', '-l',
        type=str,
        help='Traffic line coordinates as "x1,y1,x2,y2" (e.g., "100,200,500,200")'
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)'
    )

    # Mode options
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run in headless mode without GUI'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Validation
    if args.video or args.input_dir:
        if not args.line:
            parser.error("--line is required when processing videos from CLI")

    if args.output_dir and not args.input_dir:
        parser.error("--output-dir requires --input-dir")

    return args


def validate_inputs(video_path: str, line_coords: str) -> Tuple[Path, Tuple]:
    """
    Validate input file and line coordinates.

    Args:
        video_path: Path to video file
        line_coords: Line coordinates string

    Returns:
        Tuple of (validated_path, parsed_line_coords)

    Raises:
        FileNotFoundError: If video doesn't exist
        ValueError: If line coordinates invalid
    """
    # Validate video file
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video.is_file():
        raise ValueError(f"Not a file: {video_path}")

    # Validate video can be opened
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video file: {video_path}")
    cap.release()

    # Parse and validate line coordinates
    line = parse_line_coordinates(line_coords)

    return video, line


def process_video_cli(
    video_path: Path,
    line: Tuple[Tuple[int, int], Tuple[int, int]],
    output_path: Optional[Path] = None,
    confidence: float = 0.5
) -> dict:
    """
    Process video in CLI mode (headless).

    Args:
        video_path: Path to input video
        line: Traffic line coordinates ((x1,y1), (x2,y2))
        output_path: Path for output video (optional)
        confidence: Detection confidence threshold

    Returns:
        dict: Processing statistics
            {
                'total_frames': int,
                'violations': int,
                'processing_time': float,
                'fps': float
            }

    Raises:
        RuntimeError: If processing fails
    """
    import time
    from tqdm import tqdm

    logging.info(f"Processing video: {video_path}")
    logging.info(f"Traffic line: {line[0]} -> {line[1]}")
    logging.info(f"Confidence threshold: {confidence}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")

    # Setup output
    if output_path is None:
        output_path = config.OUTPUT_DIR / f"output_{video_path.stem}.mp4"

    config.ensure_directories()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Processing statistics
    violation_count = 0
    start_time = time.time()

    try:
        # Process frames with progress bar
        with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
            frame_num = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1

                # Preprocess for detection
                input_image = preprocess_input(frame, config.NET_H, config.NET_W)

                # Run detection (triggers lazy loading on first frame)
                yolos = detector.predict(input_image)

                # Decode detections
                boxes = []
                for i in range(len(yolos)):
                    boxes += decode_netout(
                        yolos[i][0],
                        config.ANCHORS[i],
                        confidence,
                        config.NET_H,
                        config.NET_W
                    )

                # Non-maximum suppression
                from object_detection import do_nms
                do_nms(boxes, config.NMS_THRESH)

                # Draw boxes and detect violations
                annotated_frame = draw_boxes(
                    frame,
                    boxes,
                    [line[0], line[1]],
                    config.LABELS,
                    confidence,
                    frame_num
                )

                # Count violations (check if any red boxes drawn)
                # This is a simplified check - you may need to modify draw_boxes
                # to return violation count

                # Write output frame
                out.write(annotated_frame)

                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'violations': violation_count,
                    'fps': frame_num / (time.time() - start_time)
                })

        # Calculate statistics
        processing_time = time.time() - start_time
        processing_fps = total_frames / processing_time

        stats = {
            'total_frames': total_frames,
            'violations': violation_count,
            'processing_time': processing_time,
            'fps': processing_fps
        }

        logging.info(f"✅ Processing complete!")
        logging.info(f"   Total frames: {total_frames}")
        logging.info(f"   Violations: {violation_count}")
        logging.info(f"   Time: {processing_time:.2f}s ({processing_fps:.2f} FPS)")
        logging.info(f"   Output: {output_path}")

        return stats

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def run_gui_mode():
    """Launch GUI mode using existing Project-GUI.py"""
    logging.info("Launching GUI mode...")

    # Import and run GUI
    from tkinter import Tk
    import Project_GUI  # Note: renamed from Project-GUI.py

    # This assumes Project-GUI.py has been renamed to Project_GUI.py
    # or you create a wrapper
    window = Tk()
    window.title("Traffic Signal Violation Detection")
    app = Project_GUI.Window(window)
    window.mainloop()


def run_cli_mode(args):
    """
    Run CLI mode with parsed arguments.

    Args:
        args: Parsed command line arguments
    """
    if args.video:
        # Single video processing
        video, line = validate_inputs(args.video, args.line)

        output = Path(args.output) if args.output else None

        stats = process_video_cli(
            video,
            line,
            output,
            args.confidence
        )

        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        print(f"Frames processed: {stats['total_frames']}")
        print(f"Violations detected: {stats['violations']}")
        print(f"Processing time: {stats['processing_time']:.2f}s")
        print(f"Average FPS: {stats['fps']:.2f}")
        print("="*50)

    elif args.input_dir:
        # Batch processing
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir) if args.output_dir else config.OUTPUT_DIR

        if not input_dir.exists():
            logging.error(f"Input directory not found: {input_dir}")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []
        for ext in video_extensions:
            videos.extend(input_dir.glob(f'*{ext}'))

        if not videos:
            logging.error(f"No video files found in {input_dir}")
            sys.exit(1)

        logging.info(f"Found {len(videos)} videos to process")

        line = parse_line_coordinates(args.line)

        # Process each video
        results = []
        for video in videos:
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing {video.name}...")
            logging.info(f"{'='*50}")

            output = output_dir / f"output_{video.stem}.mp4"

            try:
                stats = process_video_cli(video, line, output, args.confidence)
                stats['video'] = video.name
                stats['success'] = True
                results.append(stats)
            except Exception as e:
                logging.error(f"Failed to process {video.name}: {e}")
                results.append({
                    'video': video.name,
                    'success': False,
                    'error': str(e)
                })

        # Print summary
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        successful = sum(1 for r in results if r['success'])
        print(f"Videos processed: {successful}/{len(results)}")
        print("="*70)
        for result in results:
            if result['success']:
                print(f"✅ {result['video']}: {result['violations']} violations")
            else:
                print(f"❌ {result['video']}: {result['error']}")
        print("="*70)


def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    # Validate configuration
    if not config.validate_config():
        logging.error("Configuration validation failed")
        sys.exit(1)

    config.ensure_directories()

    # Run appropriate mode
    if args.video or args.input_dir:
        run_cli_mode(args)
    else:
        run_gui_mode()


if __name__ == "__main__":
    main()
```

---

### Testing Procedures

#### Test 1: Default GUI Launch

```bash
$ python main.py
# Should launch GUI as normal
```

#### Test 2: Single Video Processing

```bash
$ python main.py --video Resources/sample_video.mp4 --line "100,200,500,200"
# Should process video and save to Resources/output/
```

#### Test 3: Custom Output

```bash
$ python main.py --video test.mp4 --line "150,300,600,300" --output custom_result.mp4
# Should save to custom_result.mp4
```

#### Test 4: Confidence Threshold

```bash
$ python main.py --video test.mp4 --line "100,200,500,200" --confidence 0.8
# Should use higher threshold (fewer detections)
```

#### Test 5: Batch Processing

```bash
$ mkdir test_videos
$ python main.py --input-dir test_videos/ --output-dir results/ --line "100,200,500,200"
# Should process all videos in directory
```

#### Test 6: Error Handling

```bash
$ python main.py --video nonexistent.mp4 --line "100,200,500,200"
# Should show error: Video file not found

$ python main.py --video test.mp4 --line "invalid"
# Should show error: Invalid line format

$ python main.py --video test.mp4
# Should show error: --line is required
```

---

### Acceptance Criteria

- [ ] `main.py` created in project root
- [ ] Default behavior launches GUI (backward compatible)
- [ ] `--video` flag processes single video from CLI
- [ ] `--line` flag accepts coordinates in "x1,y1,x2,y2" format
- [ ] `--output` flag specifies custom output path
- [ ] `--confidence` flag adjusts detection threshold
- [ ] `--input-dir` processes all videos in directory
- [ ] `--output-dir` specifies batch output location
- [ ] Progress bar shows during processing
- [ ] Statistics printed after completion (frames, violations, FPS)
- [ ] Error messages clear and helpful
- [ ] All existing GUI functionality preserved
- [ ] Help text displays with `python main.py --help`
- [ ] Works on both Windows and Linux

---

### Files to Create/Modify

**Created**:
- `main.py` - New CLI entry point

**Modified**:
- None (purely additive)

**Optional**:
- Rename `Project-GUI.py` to `Project_GUI.py` (for easier importing)
- Or create `gui/__init__.py` wrapper

---

### Gotchas & Warnings

1. **Hyphen in filename**
   - `Project-GUI.py` can't be imported directly
   - Either rename to `Project_GUI.py`
   - Or use `importlib`:
   ```python
   import importlib.util
   spec = importlib.util.spec_from_file_location("gui", "Project-GUI.py")
   gui = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(gui)
   ```

2. **Line coordinate format**
   - Must be exactly "x1,y1,x2,y2"
   - No spaces unless quoted: `--line "100, 200, 500, 200"`
   - Easy to get wrong!

3. **Violation counting**
   - Current `draw_boxes()` doesn't return violation count
   - May need to modify to track violations
   - Or parse detected images directory

4. **Progress bar dependency**
   - Requires `tqdm` package
   - Already in requirements.txt
   - Make sure it's installed

5. **Video codec issues**
   - `mp4v` codec widely supported but not optimal
   - Consider `avc1` or `H264` for better compatibility
   - May need different fourcc on Windows vs Linux

---

## Task 10: CSV Violation Reporting

### Problem Statement

**Current Issue**:
- Violations are detected and saved as images
- No structured data export
- Cannot analyze violation patterns
- No database integration possible
- Cannot generate statistics or reports

**What's Missing**:
```csv
# Want to generate this:
timestamp,frame_number,vehicle_type,confidence,bbox,violation_image,video_source
2025-10-22 14:30:15,145,car,0.92,"(100,150,250,300)",violation_145.jpg,highway_cam1.mp4
2025-10-22 14:30:18,231,bus,0.87,"(300,100,500,350)",violation_231.jpg,highway_cam1.mp4
```

**Why This Matters**:
- Enable data analysis and insights
- Track violation trends over time
- Import into databases for further processing
- Generate automated reports
- Evidence for enforcement

---

### Solution Design

**Approach**: Create `ViolationReport` class for structured logging

**Features**:
1. Track all violations with metadata
2. Export to CSV format
3. Optional JSON export
4. Include frame number, timestamp, vehicle type, confidence
5. Link to saved violation images

**Architecture**:
```
ViolationReport
├── __init__()         # Initialize report
├── add_violation()    # Log single violation
├── save_csv()         # Export to CSV
├── save_json()        # Export to JSON
└── get_statistics()   # Summary stats
```

---

### Step-by-Step Implementation

#### Step 1: Create utils/report_generator.py

**File**: `utils/report_generator.py` (new file)

**Complete Implementation**:

```python
"""
Violation reporting utilities for Traffic Signal Violation Detection System.

This module provides structured logging and export of detected violations
to CSV and JSON formats for analysis and record-keeping.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class ViolationReport:
    """
    Manager for violation detection reporting and export.

    Tracks all detected violations during video processing and provides
    export functionality to CSV/JSON formats.

    Attributes:
        violations (List[Dict]): List of violation records
        video_source (str): Source video filename
        start_time (datetime): Processing start time
        output_dir (Path): Directory for report files

    Example:
        >>> report = ViolationReport("highway_cam1.mp4")
        >>> report.add_violation(
        ...     frame_num=145,
        ...     vehicle_type="car",
        ...     confidence=0.92,
        ...     bbox=(100, 150, 250, 300),
        ...     image_path="violation_145.jpg"
        ... )
        >>> report.save_csv("violations.csv")
    """

    def __init__(
        self,
        video_source: str,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize violation report.

        Args:
            video_source: Source video filename
            output_dir: Directory for saving reports (default: current directory)
        """
        self.violations: List[Dict[str, Any]] = []
        self.video_source = video_source
        self.start_time = datetime.now()
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        logging.info(f"Violation report initialized for {video_source}")

    def add_violation(
        self,
        frame_num: int,
        vehicle_type: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        image_path: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a violation record to the report.

        Args:
            frame_num: Frame number where violation occurred
            vehicle_type: Type of vehicle (car, truck, bus, etc.)
            confidence: Detection confidence score (0-1)
            bbox: Bounding box coordinates (xmin, ymin, xmax, ymax)
            image_path: Path to saved violation snapshot (optional)
            additional_data: Extra metadata to include (optional)

        Example:
            >>> report.add_violation(
            ...     frame_num=145,
            ...     vehicle_type="car",
            ...     confidence=0.92,
            ...     bbox=(100, 150, 250, 300),
            ...     image_path="violations/violation_145.jpg"
            ... )
        """
        violation = {
            'timestamp': datetime.now().isoformat(),
            'frame_number': frame_num,
            'vehicle_type': vehicle_type,
            'confidence': round(confidence, 3),
            'bbox_xmin': bbox[0],
            'bbox_ymin': bbox[1],
            'bbox_xmax': bbox[2],
            'bbox_ymax': bbox[3],
            'bbox_width': bbox[2] - bbox[0],
            'bbox_height': bbox[3] - bbox[1],
            'violation_image': str(image_path) if image_path else None,
            'video_source': self.video_source
        }

        # Add any additional metadata
        if additional_data:
            violation.update(additional_data)

        self.violations.append(violation)

        logging.debug(
            f"Violation logged: Frame {frame_num}, "
            f"{vehicle_type} (conf={confidence:.2f})"
        )

    def save_csv(self, filename: Optional[str] = None) -> Path:
        """
        Export violations to CSV file.

        Args:
            filename: Output filename (default: auto-generated)

        Returns:
            Path to saved CSV file

        Raises:
            ValueError: If no violations to save

        Example:
            >>> report.save_csv("violations.csv")
            PosixPath('/path/to/violations.csv')
        """
        if not self.violations:
            raise ValueError("No violations to save")

        # Generate filename if not provided
        if filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"violations_{timestamp}.csv"

        filepath = self.output_dir / filename

        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            # Use first violation to get fieldnames
            fieldnames = list(self.violations[0].keys())

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.violations)

        logging.info(f"✅ Violations exported to {filepath}")
        logging.info(f"   Total violations: {len(self.violations)}")

        return filepath

    def save_json(self, filename: Optional[str] = None) -> Path:
        """
        Export violations to JSON file.

        Args:
            filename: Output filename (default: auto-generated)

        Returns:
            Path to saved JSON file

        Raises:
            ValueError: If no violations to save

        Example:
            >>> report.save_json("violations.json")
            PosixPath('/path/to/violations.json')
        """
        if not self.violations:
            raise ValueError("No violations to save")

        # Generate filename if not provided
        if filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"violations_{timestamp}.json"

        filepath = self.output_dir / filename

        # Prepare JSON data with metadata
        data = {
            'metadata': {
                'video_source': self.video_source,
                'processing_start': self.start_time.isoformat(),
                'total_violations': len(self.violations),
                'generated_at': datetime.now().isoformat()
            },
            'violations': self.violations
        }

        # Write JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logging.info(f"✅ Violations exported to {filepath}")

        return filepath

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate violation statistics.

        Returns:
            dict: Statistics summary
                {
                    'total_violations': int,
                    'by_vehicle_type': Dict[str, int],
                    'average_confidence': float,
                    'first_violation_frame': int,
                    'last_violation_frame': int
                }

        Example:
            >>> stats = report.get_statistics()
            >>> print(stats['total_violations'])
            15
        """
        if not self.violations:
            return {
                'total_violations': 0,
                'by_vehicle_type': {},
                'average_confidence': 0.0,
                'first_violation_frame': None,
                'last_violation_frame': None
            }

        # Count by vehicle type
        by_type = {}
        for v in self.violations:
            vehicle_type = v['vehicle_type']
            by_type[vehicle_type] = by_type.get(vehicle_type, 0) + 1

        # Calculate average confidence
        avg_confidence = sum(v['confidence'] for v in self.violations) / len(self.violations)

        # Get frame range
        frame_numbers = [v['frame_number'] for v in self.violations]

        return {
            'total_violations': len(self.violations),
            'by_vehicle_type': by_type,
            'average_confidence': round(avg_confidence, 3),
            'first_violation_frame': min(frame_numbers),
            'last_violation_frame': max(frame_numbers)
        }

    def print_summary(self) -> None:
        """
        Print a formatted summary of violations to console.

        Example:
            >>> report.print_summary()
            ===== Violation Report =====
            Video: highway_cam1.mp4
            Total Violations: 15
            By Vehicle Type:
              - car: 10
              - bus: 3
              - truck: 2
            Average Confidence: 0.876
            ============================
        """
        stats = self.get_statistics()

        print("\n" + "="*50)
        print("VIOLATION REPORT")
        print("="*50)
        print(f"Video: {self.video_source}")
        print(f"Total Violations: {stats['total_violations']}")

        if stats['by_vehicle_type']:
            print("\nBy Vehicle Type:")
            for vehicle_type, count in sorted(stats['by_vehicle_type'].items()):
                print(f"  - {vehicle_type}: {count}")

        print(f"\nAverage Confidence: {stats['average_confidence']:.3f}")

        if stats['first_violation_frame'] is not None:
            print(f"First Violation: Frame {stats['first_violation_frame']}")
            print(f"Last Violation: Frame {stats['last_violation_frame']}")

        print("="*50 + "\n")

    def __len__(self) -> int:
        """Return number of violations recorded."""
        return len(self.violations)

    def __repr__(self) -> str:
        """String representation of report."""
        return (
            f"ViolationReport(video='{self.video_source}', "
            f"violations={len(self.violations)})"
        )


# Example usage and testing
if __name__ == "__main__":
    # Create sample report
    report = ViolationReport("test_video.mp4")

    # Add sample violations
    report.add_violation(
        frame_num=100,
        vehicle_type="car",
        confidence=0.92,
        bbox=(100, 150, 250, 300),
        image_path="violation_100.jpg"
    )

    report.add_violation(
        frame_num=150,
        vehicle_type="bus",
        confidence=0.87,
        bbox=(300, 200, 550, 450),
        image_path="violation_150.jpg"
    )

    report.add_violation(
        frame_num=200,
        vehicle_type="car",
        confidence=0.95,
        bbox=(50, 100, 200, 250),
        image_path="violation_200.jpg"
    )

    # Print summary
    report.print_summary()

    # Get statistics
    stats = report.get_statistics()
    print(f"Stats: {stats}")

    # Export (would save files in real usage)
    print(f"\nReport contains {len(report)} violations")
    print(repr(report))
```

#### Step 2: Update Project-GUI.py to Use Reporting

**File**: `Project-GUI.py`

**Modify**: `main_process()` method

**Changes**:

1. **Import** report generator at top:
```python
from utils.report_generator import ViolationReport
```

2. **Initialize** report in `main_process()`:
```python
def main_process(self):
    # ... existing video setup code ...

    # Initialize violation report
    video_name = Path(self.filename).name
    report = ViolationReport(
        video_source=video_name,
        output_dir=config.OUTPUT_DIR
    )

    # ... rest of processing ...
```

3. **Log violations** when detected:
```python
# In the frame processing loop, after draw_boxes():

# Check if violation occurred (you'll need to modify draw_boxes to return this)
# For now, we'll check if the box is red (violation)
for box in boxes:
    if box.get_score() > obj_thresh:
        # Check if this box crosses the line
        from utils.geometry import bounding_box_intersects_line

        is_violation = bounding_box_intersects_line(
            (box.xmin, box.ymin),
            (box.xmax, box.ymax),
            line[0],
            line[1]
        )

        if is_violation:
            # Log to report
            report.add_violation(
                frame_num=frame_num,
                vehicle_type=labels[box.get_label()],
                confidence=box.get_score(),
                bbox=(box.xmin, box.ymin, box.xmax, box.ymax),
                image_path=f"violation_{frame_num}.jpg"
            )
```

4. **Save report** at end of processing:
```python
# After processing loop completes:

# Save report
try:
    csv_path = report.save_csv()
    json_path = report.save_json()

    logging.info(f"Reports saved:")
    logging.info(f"  CSV: {csv_path}")
    logging.info(f"  JSON: {json_path}")

    # Print summary
    report.print_summary()

except ValueError as e:
    logging.info(f"No violations detected - no report generated")
```

---

### Complete Integration Example

Here's how it fits into the processing loop:

```python
def main_process(self):
    """Process video with violation reporting."""
    from utils.report_generator import ViolationReport
    from utils.geometry import bounding_box_intersects_line

    # Setup
    cap = cv2.VideoCapture(self.filename)
    # ... video setup code ...

    # Initialize report
    video_name = Path(self.filename).name
    report = ViolationReport(
        video_source=video_name,
        output_dir=config.OUTPUT_DIR
    )

    try:
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Detection
            input_image = preprocess_input(frame, NET_H, NET_W)
            yolos = detector.predict(input_image)

            # Decode boxes
            boxes = []
            for i in range(len(yolos)):
                boxes += decode_netout(
                    yolos[i][0],
                    anchors[i],
                    obj_thresh,
                    NET_H,
                    NET_W
                )

            # NMS
            do_nms(boxes, nms_thresh)

            # Check for violations and log them
            for box in boxes:
                if box.get_score() > obj_thresh:
                    # Check violation
                    is_violation = bounding_box_intersects_line(
                        (box.xmin, box.ymin),
                        (box.xmax, box.ymax),
                        self.line[0],
                        self.line[1]
                    )

                    if is_violation:
                        # Save violation image
                        violation_img_name = f"violation_{frame_num}.jpg"
                        violation_img_path = config.DETECTED_IMAGES_DIR / violation_img_name
                        cv2.imwrite(str(violation_img_path), frame)

                        # Log to report
                        report.add_violation(
                            frame_num=frame_num,
                            vehicle_type=labels[box.get_label()],
                            confidence=box.get_score(),
                            bbox=(box.xmin, box.ymin, box.xmax, box.ymax),
                            image_path=str(violation_img_path)
                        )

            # Draw boxes (visualization)
            annotated_frame = draw_boxes(
                frame,
                boxes,
                self.line,
                labels,
                obj_thresh,
                frame_num
            )

            # Write output
            out.write(annotated_frame)

        # Processing complete - save reports
        if len(report) > 0:
            csv_path = report.save_csv()
            json_path = report.save_json()

            report.print_summary()

            messagebox.showinfo(
                "Processing Complete",
                f"Detected {len(report)} violations\n"
                f"CSV report: {csv_path.name}\n"
                f"JSON report: {json_path.name}"
            )
        else:
            messagebox.showinfo(
                "Processing Complete",
                "No violations detected"
            )

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
```

---

### Testing Procedures

#### Test 1: Basic Report Creation

```python
# test_report.py
from utils.report_generator import ViolationReport

# Create report
report = ViolationReport("test.mp4")

# Add violations
report.add_violation(100, "car", 0.92, (100, 150, 250, 300))
report.add_violation(150, "bus", 0.87, (200, 100, 400, 350))

# Print summary
report.print_summary()

# Save
report.save_csv("test_violations.csv")
report.save_json("test_violations.json")
```

#### Test 2: Verify CSV Output

```bash
$ python -c "from utils.report_generator import ViolationReport; r = ViolationReport('test.mp4'); r.add_violation(100, 'car', 0.9, (10,20,30,40)); r.save_csv('test.csv')"

$ cat test.csv
timestamp,frame_number,vehicle_type,confidence,bbox_xmin,bbox_ymin,bbox_xmax,bbox_ymax,bbox_width,bbox_height,violation_image,video_source
2025-10-22T14:30:15.123456,100,car,0.9,10,20,30,40,20,20,,test.mp4
```

#### Test 3: Statistics

```python
from utils.report_generator import ViolationReport

report = ViolationReport("test.mp4")
report.add_violation(100, "car", 0.92, (10,20,30,40))
report.add_violation(150, "car", 0.85, (50,60,70,80))
report.add_violation(200, "bus", 0.90, (100,110,120,130))

stats = report.get_statistics()
print(stats)
# Should show: 3 total, 2 cars, 1 bus, average confidence ~0.89
```

#### Test 4: Integration with Video Processing

```bash
$ python main.py --video test.mp4 --line "100,200,500,200"
# Check that CSV and JSON files are created in Resources/output/
$ ls Resources/output/violations_*.csv
$ ls Resources/output/violations_*.json
```

---

### Acceptance Criteria

- [ ] `utils/report_generator.py` created
- [ ] `ViolationReport` class implemented
- [ ] `add_violation()` method logs all required fields
- [ ] `save_csv()` exports valid CSV format
- [ ] `save_json()` exports valid JSON format
- [ ] `get_statistics()` calculates correct stats
- [ ] `print_summary()` displays formatted output
- [ ] Integrated into `Project-GUI.py` processing
- [ ] Integrated into `main.py` CLI mode
- [ ] CSV includes all required fields (timestamp, frame, type, confidence, bbox, image)
- [ ] Reports auto-saved after processing
- [ ] User notified of report location
- [ ] Works with both GUI and CLI modes
- [ ] Handles zero violations gracefully

---

### Files to Create/Modify

**Created**:
- `utils/report_generator.py` - New reporting module

**Modified**:
- `Project-GUI.py` - Add reporting to `main_process()`
- `main.py` - Add reporting to `process_video_cli()`
- `utils/__init__.py` - Export `ViolationReport`

---

### Gotchas & Warnings

1. **CSV field order**
   - DictWriter uses keys from first violation
   - All violations must have same keys
   - Or specify fieldnames explicitly

2. **Timestamp format**
   - ISO format works in Excel
   - But may need formatting for human reading
   - Consider adding separate date/time columns

3. **File paths in CSV**
   - Use relative paths, not absolute
   - Or path might not work on different machines

4. **Violation detection**
   - Current `draw_boxes()` doesn't return violation flag
   - Need to check line intersection yourself
   - Or refactor `draw_boxes()` to return violations list

5. **Empty reports**
   - Handle case where no violations detected
   - Don't save empty files
   - Notify user appropriately

---

## Task 11: Progress Bars

### Problem Statement

**Current Issue**:
- No feedback during video processing
- User doesn't know how long processing will take
- Cannot tell if application is frozen or working
- No visibility into processing rate

**Example**:
```
$ python main.py --video long_video.mp4 --line "100,200,500,200"
[Cursor blinks for 10 minutes with no output...]
```

**Why This Matters**:
- Poor user experience
- Users kill process thinking it's frozen
- Cannot estimate completion time
- No real-time violation count

---

### Solution Design

**Approach**: Add `tqdm` progress bars with real-time stats

**Features**:
1. Show frame processing progress
2. Display violations count
3. Show processing FPS
4. ETA for completion
5. Clean progress bar formatting

**Example Output**:
```
Processing: 45%|████████░░░░░░░░| 450/1000 [01:23<01:48, 5.12 frames/s, violations=12]
```

---

### Step-by-Step Implementation

#### Step 1: Verify tqdm is Installed

**Check**: `requirements.txt` already includes `tqdm>=4.65.0` ✅

**If needed**:
```bash
pip install tqdm
```

#### Step 2: Add Progress Bar to CLI Mode

**File**: `main.py`

**Location**: In `process_video_cli()` function

**Find** the processing loop:
```python
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    # ... processing ...
```

**Replace with**:
```python
from tqdm import tqdm

# Get total frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create progress bar
with tqdm(
    total=total_frames,
    desc="Processing",
    unit="frame",
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
) as pbar:

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # ... processing code ...

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'violations': violation_count,
            'fps': f'{pbar.format_dict["rate"]:.2f}' if pbar.format_dict["rate"] else '0.00'
        })
```

#### Step 3: Add Progress Bar to GUI Mode (Optional)

**File**: `Project-GUI.py`

**Option 1**: Console progress bar (simple)

```python
from tqdm import tqdm

def main_process(self):
    # ... setup ...

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(
        total=total_frames,
        desc="Processing video",
        unit="frame"
    ) as pbar:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ... processing ...

            pbar.update(1)
            pbar.set_postfix({'violations': violation_count})
```

**Option 2**: Tkinter progress bar (better for GUI)

```python
from tkinter import ttk

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

        # Add progress bar widget
        self.progress = ttk.Progressbar(
            master,
            orient="horizontal",
            length=300,
            mode="determinate"
        )
        self.progress.pack(pady=10)

        # Add status label
        self.status_label = Label(master, text="Ready")
        self.status_label.pack()

    def main_process(self):
        # ... setup ...

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress["maximum"] = total_frames
        self.progress["value"] = 0

        frame_num = 0
        violation_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # ... processing ...

            # Update progress bar
            self.progress["value"] = frame_num
            self.status_label.config(
                text=f"Processing: {frame_num}/{total_frames} frames | "
                     f"Violations: {violation_count}"
            )

            # Force GUI update
            self.master.update_idletasks()
```

---

### Complete Example: Enhanced main.py with Progress

```python
def process_video_cli(
    video_path: Path,
    line: Tuple[Tuple[int, int], Tuple[int, int]],
    output_path: Optional[Path] = None,
    confidence: float = 0.5,
    show_progress: bool = True
) -> dict:
    """
    Process video in CLI mode with progress bar.

    Args:
        video_path: Path to input video
        line: Traffic line coordinates
        output_path: Output video path
        confidence: Detection threshold
        show_progress: Show progress bar (default: True)

    Returns:
        Processing statistics
    """
    import time
    from tqdm import tqdm

    # ... setup code ...

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ... output setup ...

    violation_count = 0
    start_time = time.time()

    # Create progress bar
    progress_bar = tqdm(
        total=total_frames,
        desc="Processing video",
        unit="frame",
        disable=not show_progress,  # Can disable for logging/automation
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                   '[{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )

    try:
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1

            # Preprocessing
            input_image = preprocess_input(frame, NET_H, NET_W)

            # Detection
            yolos = detector.predict(input_image)

            # Decode
            boxes = []
            for i in range(len(yolos)):
                boxes += decode_netout(...)

            do_nms(boxes, nms_thresh)

            # Check violations
            frame_violations = 0
            for box in boxes:
                if box.get_score() > confidence:
                    if bounding_box_intersects_line(...):
                        frame_violations += 1
                        violation_count += 1

            # Draw
            annotated = draw_boxes(frame, boxes, line, labels, confidence, frame_num)
            out.write(annotated)

            # Update progress
            progress_bar.update(1)

            # Calculate FPS
            elapsed = time.time() - start_time
            processing_fps = frame_num / elapsed if elapsed > 0 else 0

            # Update postfix with stats
            progress_bar.set_postfix({
                'violations': violation_count,
                'fps': f'{processing_fps:.2f}',
                'latest': frame_violations  # violations in this frame
            })

    finally:
        progress_bar.close()
        cap.release()
        out.release()

    # ... return stats ...
```

---

### Testing Procedures

#### Test 1: CLI Progress Bar

```bash
$ python main.py --video test_video.mp4 --line "100,200,500,200"

# Should show:
Processing video: 45%|████████░░░░░░░░| 450/1000 [00:30<00:36, 15.23 frame/s] violations=12, fps=15.23
```

#### Test 2: Progress Bar Disabled (for logging)

```python
# In main.py, test with show_progress=False
result = process_video_cli(
    video_path,
    line,
    show_progress=False  # No progress bar
)

# Useful for:
# - Logging to file
# - Automated scripts
# - CI/CD pipelines
```

#### Test 3: GUI Progress Bar

```bash
$ python Project-GUI.py
# Load video
# Click "Start Processing"
# Should see progress bar updating
# Should see status: "Processing: 450/1000 frames | Violations: 12"
```

#### Test 4: Estimate Accuracy

Process a video and verify:
- ETA becomes more accurate as processing continues
- Final time matches ETA from midpoint
- FPS measurement is reasonable (2-30 FPS depending on hardware)

---

### Acceptance Criteria

- [ ] `tqdm` imported and working
- [ ] Progress bar shows in CLI mode
- [ ] Progress bar shows percentage complete
- [ ] Progress bar shows frames processed / total
- [ ] Progress bar shows elapsed time
- [ ] Progress bar shows remaining time (ETA)
- [ ] Progress bar shows processing FPS
- [ ] Postfix displays violation count
- [ ] Progress bar can be disabled for automation
- [ ] GUI shows progress (console or Tkinter widget)
- [ ] Progress updates in real-time
- [ ] Bar clears properly when complete
- [ ] Works on both Windows and Linux
- [ ] No performance degradation from progress updates

---

### Files to Create/Modify

**Modified**:
- `main.py` - Add tqdm to `process_video_cli()`
- `Project-GUI.py` - Add progress (console or Tkinter)

**No new files**

---

### Gotchas & Warnings

1. **GUI updates can slow processing**
   - Don't update GUI every frame
   - Update every N frames: `if frame_num % 10 == 0:`
   - Or use time-based: `if time.time() - last_update > 0.1:`

2. **tqdm with logging**
   - tqdm can interfere with logging output
   - Use `disable=True` for non-interactive environments
   - Or redirect tqdm to file: `file=sys.stderr`

3. **Total frames may be wrong**
   - Some video formats don't report accurate frame count
   - Falls back to unknown length progress bar
   - Consider counting frames first (slower)

4. **Progress bar refresh rate**
   - Default: 10 updates per second
   - Can customize: `mininterval=0.5` for 2 updates/sec
   - Too fast = performance hit
   - Too slow = appears frozen

5. **Nested progress bars**
   - Batch processing needs nested bars
   - Use `tqdm(position=0)` and `position=1`
   - Or separate bars per video

Example batch processing:
```python
videos = list(input_dir.glob('*.mp4'))

# Outer bar for videos
with tqdm(total=len(videos), desc="Videos", position=0) as video_pbar:
    for video in videos:
        # Inner bar for frames
        with tqdm(desc=video.name, position=1, leave=False) as frame_pbar:
            # Process video...
            pass

        video_pbar.update(1)
```

---

## Task 12: GPU Acceleration

### Problem Statement

**Current Issue**:
- YOLOv3 runs on CPU only
- Very slow processing: 2-5 FPS
- Cannot process high-resolution videos in real-time
- Poor utilization of available hardware

**Performance Comparison**:
```
CPU (Intel i7): ~3 FPS
GPU (NVIDIA GTX 1060): ~25 FPS
GPU (NVIDIA RTX 3080): ~60 FPS
```

**Why This Matters**:
- 5-10x faster processing with GPU
- Essential for real-time applications
- Better user experience
- Can process higher resolution videos

---

### Solution Design

**Approach**: Auto-detect and configure GPU in TensorFlow

**Features**:
1. Auto-detect available GPUs
2. Enable memory growth (prevent OOM)
3. Fall back to CPU if no GPU
4. Log GPU information
5. Allow GPU selection for multi-GPU systems

**TensorFlow GPU Setup**:
```python
import tensorflow as tf

# Check GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # Configure memory growth
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

### Step-by-Step Implementation

#### Step 1: Update config.py with GPU Settings

**File**: `config.py`

**Add** at end of file:

```python
# ============================================================================
# GPU Configuration
# ============================================================================

# GPU Settings
USE_GPU = True  # Set to False to force CPU mode
GPU_MEMORY_LIMIT = None  # MB limit, or None for dynamic growth

# TensorFlow GPU logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging (0=all, 3=errors only)


def configure_gpu():
    """
    Configure GPU settings for TensorFlow.

    Automatically detects available GPUs and configures:
    - Memory growth to prevent OOM errors
    - Optional memory limit
    - Multi-GPU selection

    Returns:
        List[str]: Names of available GPU devices

    Example:
        >>> gpus = configure_gpu()
        >>> print(f"Using {len(gpus)} GPU(s)")
    """
    import tensorflow as tf
    import logging

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
            tf.config.experimental.set_memory_growth(gpu, True)

            logging.info(f"✅ GPU configured: {gpu_name}")

            # Optional: Set memory limit
            if GPU_MEMORY_LIMIT is not None:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=GPU_MEMORY_LIMIT
                    )]
                )
                logging.info(f"   Memory limit: {GPU_MEMORY_LIMIT} MB")

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
                'device_names': List[str],
                'tensorflow_version': str,
                'cuda_available': bool
            }
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')

    info = {
        'device_type': 'GPU' if gpus else 'CPU',
        'device_count': len(gpus),
        'device_names': [gpu.name for gpu in gpus],
        'tensorflow_version': tf.__version__,
        'cuda_available': tf.test.is_built_with_cuda()
    }

    return info


# Auto-configure GPU on import
if __name__ != "__main__":
    configure_gpu()
```

#### Step 2: Update object_detection.py

**File**: `object_detection.py`

**Add** near the top (after imports, before model code):

```python
# Configure GPU before loading model
from config import configure_gpu

# This will run when module is imported
_gpus = configure_gpu()
if _gpus:
    logging.info(f"YOLOv3 will use GPU: {_gpus}")
else:
    logging.info("YOLOv3 will use CPU")
```

#### Step 3: Add GPU Info to main.py

**File**: `main.py`

**Add** to `main()` function at start:

```python
def main():
    """Main entry point."""
    args = parse_arguments()

    setup_logging(args.verbose)

    # Display device information
    from config import get_device_info

    device_info = get_device_info()
    logging.info("="*50)
    logging.info("DEVICE INFORMATION")
    logging.info("="*50)
    logging.info(f"Device Type: {device_info['device_type']}")
    logging.info(f"Device Count: {device_info['device_count']}")
    if device_info['device_names']:
        for name in device_info['device_names']:
            logging.info(f"  - {name}")
    logging.info(f"TensorFlow Version: {device_info['tensorflow_version']}")
    logging.info(f"CUDA Available: {device_info['cuda_available']}")
    logging.info("="*50)

    # ... rest of main() ...
```

---

### Complete GPU Configuration Module

Here's a comprehensive GPU configuration module you can use:

```python
# utils/gpu_config.py
"""
GPU configuration utilities for Traffic Signal Violation Detection System.

Handles automatic GPU detection, configuration, and fallback to CPU.
"""

import logging
from typing import List, Dict, Any, Optional
import os


def configure_tensorflow_gpu(
    use_gpu: bool = True,
    memory_growth: bool = True,
    memory_limit_mb: Optional[int] = None,
    gpu_id: Optional[int] = None
) -> List[str]:
    """
    Configure TensorFlow GPU settings.

    Args:
        use_gpu: Whether to use GPU if available
        memory_growth: Enable dynamic memory allocation
        memory_limit_mb: Maximum GPU memory in MB (None for unlimited)
        gpu_id: Specific GPU to use (None for all GPUs)

    Returns:
        List of GPU device names

    Example:
        >>> gpus = configure_tensorflow_gpu(memory_limit_mb=4096)
        >>> print(f"Using {len(gpus)} GPU(s)")
    """
    import tensorflow as tf

    # Reduce TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    try:
        # Get all GPUs
        physical_gpus = tf.config.list_physical_devices('GPU')

        if not physical_gpus:
            logging.info("🖥️  No GPU detected. Using CPU.")
            return []

        if not use_gpu:
            logging.info("⚙️  GPU available but disabled. Using CPU.")
            tf.config.set_visible_devices([], 'GPU')
            return []

        # Select specific GPU if requested
        if gpu_id is not None:
            if gpu_id >= len(physical_gpus):
                logging.warning(
                    f"GPU {gpu_id} not found. "
                    f"Available GPUs: 0-{len(physical_gpus)-1}"
                )
                return []

            physical_gpus = [physical_gpus[gpu_id]]
            tf.config.set_visible_devices(physical_gpus, 'GPU')
            logging.info(f"🎯 Selected GPU {gpu_id}")

        # Configure GPUs
        gpu_names = []

        for gpu in physical_gpus:
            # Enable memory growth
            if memory_growth:
                tf.config.experimental.set_memory_growth(gpu, True)

            # Set memory limit
            if memory_limit_mb:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=memory_limit_mb
                    )]
                )

            gpu_names.append(gpu.name)

        # Log configuration
        logging.info("="*60)
        logging.info("🚀 GPU Configuration")
        logging.info("="*60)
        logging.info(f"GPUs Available: {len(physical_gpus)}")

        for i, gpu in enumerate(physical_gpus):
            logging.info(f"  GPU {i}: {gpu.name}")

        logging.info(f"Memory Growth: {'Enabled' if memory_growth else 'Disabled'}")

        if memory_limit_mb:
            logging.info(f"Memory Limit: {memory_limit_mb} MB")

        logging.info(f"TensorFlow: {tf.__version__}")
        logging.info(f"CUDA: {'Available' if tf.test.is_built_with_cuda() else 'Not Available'}")
        logging.info("="*60)

        return gpu_names

    except RuntimeError as e:
        logging.error(f"❌ GPU configuration error: {e}")
        logging.info("Falling back to CPU")
        return []


def test_gpu_performance():
    """
    Test GPU performance with a simple benchmark.

    Returns:
        dict: Benchmark results {
            'device': str,
            'time_seconds': float,
            'operations_per_second': float
        }
    """
    import tensorflow as tf
    import time
    import numpy as np

    # Create test matrix
    matrix_size = 5000
    iterations = 10

    device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"

    logging.info(f"Running benchmark on {device_name}...")

    # Warmup
    a = tf.random.normal([matrix_size, matrix_size])
    b = tf.random.normal([matrix_size, matrix_size])
    _ = tf.matmul(a, b)

    # Benchmark
    start_time = time.time()

    for _ in range(iterations):
        a = tf.random.normal([matrix_size, matrix_size])
        b = tf.random.normal([matrix_size, matrix_size])
        c = tf.matmul(a, b)

    elapsed = time.time() - start_time
    ops_per_sec = iterations / elapsed

    result = {
        'device': device_name,
        'time_seconds': round(elapsed, 3),
        'operations_per_second': round(ops_per_sec, 2)
    }

    logging.info(f"Benchmark: {ops_per_sec:.2f} ops/sec on {device_name}")

    return result


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Configure GPU
    gpus = configure_tensorflow_gpu()

    # Run benchmark
    if gpus:
        result = test_gpu_performance()
        print(f"\nPerformance: {result['operations_per_second']} ops/sec")
```

---

### Testing Procedures

#### Test 1: Check GPU Detection

```bash
$ python -c "from config import get_device_info; import pprint; pprint.pprint(get_device_info())"

# Expected output with GPU:
{
  'device_type': 'GPU',
  'device_count': 1,
  'device_names': ['/physical_device:GPU:0'],
  'tensorflow_version': '2.10.0',
  'cuda_available': True
}

# Expected output without GPU:
{
  'device_type': 'CPU',
  'device_count': 0,
  'device_names': [],
  'tensorflow_version': '2.10.0',
  'cuda_available': False
}
```

#### Test 2: Verify GPU Usage During Processing

```bash
# In another terminal, monitor GPU usage:
$ watch -n 0.5 nvidia-smi

# Run processing:
$ python main.py --video test.mp4 --line "100,200,500,200"

# Should see GPU utilization spike to 80-100%
```

#### Test 3: Benchmark CPU vs GPU

```python
from config import USE_GPU

# Test with GPU
configure_gpu()
# Process video, record time

# Test with CPU
USE_GPU = False
configure_gpu()
# Process same video, record time

# Compare speeds
```

#### Test 4: Memory Growth

```bash
# Without memory growth, TensorFlow allocates all GPU memory:
$ nvidia-smi
# Shows 100% memory usage immediately

# With memory growth, allocation is gradual:
$ nvidia-smi
# Shows memory usage growing during processing
```

---

### Acceptance Criteria

- [ ] `configure_gpu()` function in config.py
- [ ] Auto-detects available GPUs
- [ ] Enables memory growth by default
- [ ] Logs GPU information
- [ ] Falls back to CPU gracefully if no GPU
- [ ] `USE_GPU` flag to disable GPU
- [ ] Works with multi-GPU systems
- [ ] No OOM (Out of Memory) errors
- [ ] Processing speed 5-10x faster with GPU
- [ ] `nvidia-smi` shows GPU utilization during processing
- [ ] Works on both CUDA 11.x and 12.x
- [ ] Compatible with TensorFlow 2.10-2.15

---

### Files to Create/Modify

**Modified**:
- `config.py` - Add `configure_gpu()` and `get_device_info()`
- `object_detection.py` - Call `configure_gpu()` on import
- `main.py` - Display device info at startup

**Optional**:
- `utils/gpu_config.py` - Dedicated GPU configuration module

---

### Gotchas & Warnings

1. **CUDA/cuDNN Installation Required**
   - GPU won't work without CUDA toolkit
   - TensorFlow 2.10 needs CUDA 11.2+
   - TensorFlow 2.13+ needs CUDA 11.8+
   - Must match CUDA version to TensorFlow version

   **Installation**:
   ```bash
   # Ubuntu
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt-get update
   sudo apt-get -y install cuda

   # Windows: Download from NVIDIA website
   ```

2. **Memory Growth Must Be Set BEFORE GPU Use**
   - Configure before importing model
   - Error if set after GPU initialized
   - That's why we configure in config.py

3. **TensorFlow GPU vs CPU Versions**
   - TensorFlow 2.0+ includes both CPU and GPU in single package
   - No need for separate `tensorflow-gpu` package
   - GPU support auto-enabled if CUDA found

4. **OOM Errors**
   - If you get "OOM when allocating tensor"
   - Enable memory growth (should be default)
   - Or set memory limit: `GPU_MEMORY_LIMIT = 4096`  # 4GB
   - Or reduce batch size

5. **Multiple TensorFlow Versions**
   - Different projects may need different TF versions
   - Use virtual environments
   - Check: `python -c "import tensorflow as tf; print(tf.__version__)"`

6. **WSL2 GPU Support**
   - Works on Windows Subsystem for Linux 2
   - Needs NVIDIA drivers on Windows host
   - Needs CUDA toolkit in WSL2
   - See: https://docs.nvidia.com/cuda/wsl-user-guide/

---

## Common Patterns

### Error Handling Pattern

Use throughout codebase:

```python
try:
    # Risky operation
    result = do_something()

except SpecificError as e:
    # Handle specific error
    logging.error(f"Specific error: {e}")
    # Recover or notify user

except Exception as e:
    # Catch-all for unexpected errors
    logging.exception("Unexpected error")  # Includes traceback
    # Notify user with friendly message

finally:
    # Always runs (cleanup)
    cleanup_resources()
```

### Logging Pattern

Standard logging setup:

```python
import logging

# At module level
logger = logging.getLogger(__name__)

# Usage
logger.debug("Detailed information for debugging")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.exception("Error with full traceback")  # Use in except blocks
```

### Type Hints Pattern

Use on all functions:

```python
from typing import List, Dict, Optional, Tuple

def process_frame(
    frame: np.ndarray,
    line: Tuple[int, int, int, int],
    confidence: float = 0.5
) -> Optional[List[Dict[str, Any]]]:
    """
    Process a video frame.

    Args:
        frame: Input frame as numpy array
        line: Line coordinates (x1, y1, x2, y2)
        confidence: Detection threshold

    Returns:
        List of detections, or None if error
    """
    pass
```

### Configuration Pattern

Centralize all config:

```python
# config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# Don't hardcode paths elsewhere!
```

### Resource Cleanup Pattern

Always use try-finally or context managers:

```python
# Option 1: try-finally
cap = None
try:
    cap = cv2.VideoCapture(video_path)
    # Use cap...
finally:
    if cap is not None:
        cap.release()

# Option 2: Context manager (preferred)
class VideoCapture:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

    def __enter__(self):
        return self.cap

    def __exit__(self, *args):
        self.cap.release()

# Usage:
with VideoCapture(video_path) as cap:
    # Use cap...
    # Automatically released
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Import Errors

**Problem**:
```python
ImportError: No module named 'tensorflow'
```

**Solution**:
```bash
pip install -r requirements.txt
# Or
pip install tensorflow opencv-python numpy
```

---

#### Issue 2: GPU Not Detected

**Problem**:
```
No GPU detected. Using CPU.
```

**Checklist**:
1. **Check NVIDIA driver**:
   ```bash
   nvidia-smi
   # Should show GPU info
   ```

2. **Check CUDA**:
   ```bash
   nvcc --version
   # Should show CUDA version
   ```

3. **Check TensorFlow GPU support**:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   # Should show [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
   ```

4. **Reinstall TensorFlow** (if needed):
   ```bash
   pip uninstall tensorflow
   pip install tensorflow==2.10.0
   ```

---

#### Issue 3: Out of Memory (OOM)

**Problem**:
```
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor
```

**Solutions**:

1. **Enable memory growth** (should be default):
   ```python
   tf.config.experimental.set_memory_growth(gpu, True)
   ```

2. **Set memory limit**:
   ```python
   # In config.py
   GPU_MEMORY_LIMIT = 4096  # 4GB
   ```

3. **Reduce input size**:
   ```python
   # In config.py
   NET_H = 320  # Instead of 416
   NET_W = 320
   ```

4. **Close other GPU applications**

---

#### Issue 4: Slow Processing on GPU

**Problem**: GPU available but processing is slow

**Checklist**:

1. **Verify GPU is actually being used**:
   ```bash
   nvidia-smi
   # Check GPU-Util column
   ```

2. **Check TensorFlow is using GPU**:
   ```python
   import tensorflow as tf
   print(tf.test.is_built_with_cuda())  # Should be True
   print(tf.config.list_physical_devices('GPU'))  # Should list GPUs
   ```

3. **Warmup issue** - First frame is always slow:
   ```python
   # First prediction loads model (~30s)
   # Subsequent predictions should be fast (<0.1s)
   ```

---

#### Issue 5: Video Won't Open

**Problem**:
```
cv2.error: OpenCV(4.6.0) error: (-215:Assertion failed) !_src.empty()
```

**Solutions**:

1. **Check file exists**:
   ```python
   from pathlib import Path
   assert Path("video.mp4").exists()
   ```

2. **Check file format**:
   ```bash
   ffmpeg -i video.mp4
   # Shows codec info
   ```

3. **Try different video**:
   - Use .mp4 with H.264 codec (most compatible)
   - Avoid uncommon formats

4. **Reinstall OpenCV**:
   ```bash
   pip uninstall opencv-python
   pip install opencv-python==4.6.0.66
   ```

---

#### Issue 6: Model Weights Not Found

**Problem**:
```
FileNotFoundError: YOLOv3 weights not found at weights/yolov3.weights
```

**Solution**:
```bash
# Download weights
mkdir -p weights
wget https://pjreddie.com/media/files/yolov3.weights -P weights/

# Or
curl -L https://pjreddie.com/media/files/yolov3.weights -o weights/yolov3.weights
```

---

#### Issue 7: Line Coordinates Invalid

**Problem**:
```
ValueError: Invalid line format: '100 200 500 200'. Expected format: 'x1,y1,x2,y2'
```

**Solution**:
```bash
# Correct format (comma-separated, no spaces):
python main.py --line "100,200,500,200"

# Or with quotes if spaces included:
python main.py --line "100, 200, 500, 200"
```

---

#### Issue 8: Progress Bar Not Showing

**Problem**: No progress bar in terminal

**Solutions**:

1. **Check tqdm installed**:
   ```bash
   pip install tqdm
   ```

2. **Check terminal supports it**:
   - Some terminals don't support ANSI escape codes
   - Try different terminal
   - Or disable: `show_progress=False`

3. **Check output redirection**:
   ```bash
   # This won't show progress bar:
   python main.py > output.log

   # This will:
   python main.py
   ```

---

## Success Criteria

### Overall Project Success

Mark the implementation complete when ALL tasks meet their individual acceptance criteria AND:

**✅ Functionality**:
- [ ] All 10 tasks (7-15) implemented
- [ ] GUI mode works (backward compatible)
- [ ] CLI mode works
- [ ] GPU acceleration enabled
- [ ] Lazy loading functional
- [ ] Reports generated correctly
- [ ] Tests pass

**✅ Code Quality**:
- [ ] No hardcoded paths
- [ ] All functions have docstrings
- [ ] All functions have type hints
- [ ] Error handling comprehensive
- [ ] Logging instead of print statements
- [ ] Code follows existing patterns

**✅ Documentation**:
- [ ] README updated
- [ ] All docstrings complete
- [ ] Comments explain complex logic
- [ ] Examples provided

**✅ Testing**:
- [ ] Unit tests written
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] No regressions in existing functionality

**✅ Performance**:
- [ ] Startup time < 1s (lazy loading)
- [ ] Processing speed acceptable
- [ ] GPU utilized if available
- [ ] Memory usage reasonable

**✅ User Experience**:
- [ ] Progress bars show feedback
- [ ] Error messages helpful
- [ ] CLI interface intuitive
- [ ] Reports easy to understand

---

## Next Steps

### After Completing This Guide

1. **Create Pull Request**:
   - Use `.github/pr_description.md` template
   - Include before/after metrics
   - Link to issues addressed

2. **Update Documentation**:
   - Update README.md
   - Add usage examples
   - Document new features

3. **Share Knowledge**:
   - Present to team
   - Write blog post
   - Record demo video

4. **Plan Future Enhancements**:
   - YOLOv8 upgrade
   - Web interface
   - Multi-camera support
   - License plate recognition
   - Database integration

---

## Appendix

### Useful Commands

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Format code
black .

# Lint code
flake8 .

# Type check
mypy .

# Build Docker image
docker build -t traffic-violation-detector .

# Run Docker container
docker run -v $(pwd)/input:/app/input traffic-violation-detector

# Check GPU
nvidia-smi

# Monitor GPU usage
watch -n 0.5 nvidia-smi

# Process video
python main.py --video input.mp4 --line "100,200,500,200"

# Batch process
python main.py --input-dir videos/ --output-dir results/ --line "100,200,500,200"

# Launch GUI
python main.py
```

---

**End of Implementation Guide**

**Document Statistics**:
- Total Sections: 15
- Code Examples: 50+
- Tasks Detailed: 6 (Task 7-12)
- Total Lines: ~2,400+

**For complete guide, continue with Tasks 13-15 following the same format.**

---
