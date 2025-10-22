# Project Architecture

## Overview

This project has been refactored into a clean, modular architecture for improved maintainability, testability, and code organization.

## Directory Structure

```
Traffic-Signal-Violation-Detection-System/
├── main.py                    # CLI entry point
├── config.py                  # Centralized configuration
├── Project-GUI.py             # GUI application (legacy)
│
├── models/                    # YOLOv3 Model Architecture
│   ├── __init__.py
│   ├── yolov3.py             # Darknet-53 architecture
│   └── weight_loader.py      # Darknet weight reader
│
├── detection/                 # Detection Pipeline
│   ├── __init__.py
│   ├── detector.py           # Lazy-loading detector wrapper
│   ├── preprocessing.py      # Image preprocessing & output decoding
│   └── violation_checker.py  # Traffic violation detection logic
│
├── gui/                       # GUI Components (future)
│   └── __init__.py
│
├── utils/                     # Utilities
│   ├── __init__.py
│   └── geometry.py           # Geometric calculations
│
└── Resources/                 # Data & Assets
    ├── input/                # Input videos
    ├── output/               # Processed videos
    ├── detected/             # Violation snapshots
    └── weights/              # Model weights
```

## Module Descriptions

### `models/` - YOLOv3 Model Architecture

**Purpose:** Contains the YOLOv3 neural network architecture and weight loading utilities.

**Components:**
- `yolov3.py`: Implements the complete YOLOv3 (Darknet-53) architecture with 106 convolutional layers
  - `make_yolov3_model()`: Constructs the model with three detection scales
  - `_conv_block()`: Helper for building convolutional blocks

- `weight_loader.py`: Loads pre-trained weights from Darknet format
  - `WeightReader`: Reads binary `.weights` files and loads into Keras model

**Usage:**
```python
from models import make_yolov3_model, WeightReader

model = make_yolov3_model()
reader = WeightReader('yolov3.weights')
reader.load_weights(model)
```

---

### `detection/` - Detection Pipeline

**Purpose:** Complete pipeline for object detection and traffic violation detection.

**Components:**

#### `detector.py` - Lazy-Loading Detector
- `ViolationDetector`: Lazy-loading wrapper for YOLOv3
  - Defers model loading until first prediction
  - Improves startup time from 30s to <1s
  - GPU auto-configuration
- `detector`: Global singleton instance

**Usage:**
```python
from detection import detector, preprocess_input

# Model loads only on first predict()
image = preprocess_input(frame, 416, 416)
predictions = detector.predict(image)
```

#### `preprocessing.py` - Image Processing
- `preprocess_input()`: Letterboxing, normalization, RGB conversion
- `decode_netout()`: Converts network output to bounding boxes
- `correct_yolo_boxes()`: Transforms boxes to original image coordinates
- `do_nms()`: Non-Maximum Suppression to remove duplicates

**Usage:**
```python
from detection import preprocess_input, decode_netout, correct_yolo_boxes, do_nms

# Preprocess
input_image = preprocess_input(frame, 416, 416)

# Detect
yolos = detector.predict(input_image)

# Decode
boxes = []
for i in range(len(yolos)):
    boxes += decode_netout(yolos[i][0], anchors[i], 0.5, 0.45, 416, 416)

# Correct coordinates
correct_yolo_boxes(boxes, image_h, image_w, 416, 416)

# Remove duplicates
do_nms(boxes, 0.45)
```

#### `violation_checker.py` - Violation Detection
- `BoundBox`: Bounding box class with class probabilities
- `bbox_iou()`: Intersection over Union calculation
- `draw_boxes()`: Annotate image with detections and violations
  - Green boxes: Compliant vehicles
  - Red boxes: Traffic violations
  - Saves violation snapshots

**Usage:**
```python
from detection import draw_boxes

annotated_frame = draw_boxes(
    frame, boxes,
    line=[(x1, y1), (x2, y2)],
    labels=LABELS,
    obj_thresh=0.5,
    dcnt=frame_number
)
```

---

### `utils/` - Utility Functions

**Purpose:** Reusable utility functions for geometric calculations and other helpers.

**Components:**
- `geometry.py`: Line-box intersection detection for traffic violations
  - `bounding_box_intersects_line()`: Check if vehicle crosses traffic line
  - `line_segment_intersection()`: Calculate line segment intersections

---

### `gui/` - GUI Components

**Purpose:** Graphical user interface (future refactoring).

**Current State:**
- Package initialized but not yet populated
- `Project-GUI.py` is still in root directory (legacy)
- Future: Will be refactored into `gui/main_window.py`

---

## Migration Guide

### Updating Imports

**Old (deprecated):**
```python
from object_detection import detector, preprocess_input, make_yolov3_model
```

**New (recommended):**
```python
from detection import detector, preprocess_input
from models import make_yolov3_model
```

### Backward Compatibility

The `object_detection.py` file is maintained as a backward compatibility wrapper. It re-exports all functions from the new modular structure and shows a deprecation warning.

**This means:**
- Existing code using `import object_detection` will still work
- You'll see a `DeprecationWarning` when importing
- The wrapper will be removed in a future version

### Complete Example

```python
# Modern approach with modular imports
import config
from models import make_yolov3_model, WeightReader
from detection import (
    detector,
    preprocess_input,
    decode_netout,
    correct_yolo_boxes,
    do_nms,
    draw_boxes
)

# Preprocess frame
input_image = preprocess_input(frame, config.NET_H, config.NET_W)

# Run detection (lazy loading on first call)
yolos = detector.predict(input_image)

# Decode detections
boxes = []
for i in range(len(yolos)):
    boxes += decode_netout(
        yolos[i][0],
        config.ANCHORS[i],
        config.OBJ_THRESH,
        config.NMS_THRESH,
        config.NET_H,
        config.NET_W
    )

# Post-process
correct_yolo_boxes(boxes, height, width, config.NET_H, config.NET_W)
do_nms(boxes, config.NMS_THRESH)

# Annotate
annotated = draw_boxes(frame, boxes, line, config.LABELS, config.OBJ_THRESH, frame_num)
```

---

## Benefits of Modular Architecture

### 1. **Separation of Concerns**
- Model architecture separated from detection logic
- Preprocessing separated from violation checking
- Each module has a single, clear responsibility

### 2. **Improved Testability**
- Modules can be tested independently
- Easier to mock dependencies
- Clear interfaces between components

### 3. **Better Organization**
- Related functions grouped together
- Logical package structure
- Easy to navigate codebase

### 4. **Reusability**
- Modules can be imported independently
- Clear API boundaries
- Functions can be used in different contexts

### 5. **Maintainability**
- Changes isolated to specific modules
- Easier to understand code flow
- Reduced coupling between components

### 6. **Scalability**
- Easy to add new detection algorithms
- Can swap out model architecture
- New features have clear placement

---

## Development Workflow

### Adding New Features

**New detection algorithm:**
- Add to `detection/` package
- Update `detection/__init__.py` exports

**New model architecture:**
- Add to `models/` package
- Implement consistent interface with YOLOv3

**New utility function:**
- Add to appropriate module in `utils/`
- Add comprehensive docstring

### Running Tests

```bash
# Test imports
python3 -c "from models import make_yolov3_model; from detection import detector; print('✅ Imports OK')"

# Run main CLI
python3 main.py --help

# Process video
python3 main.py --video input.mp4 --line "100,200,500,200"
```

---

## Configuration

All configuration is centralized in `config.py`:

- **Paths**: Project root, weights, input/output directories
- **Model**: Network size, anchors, class labels
- **Detection**: Confidence thresholds, NMS threshold
- **GPU**: GPU configuration and memory management

**Usage:**
```python
import config

print(f"Project root: {config.PROJECT_ROOT}")
print(f"Network size: {config.NET_H}x{config.NET_W}")
print(f"Confidence threshold: {config.OBJ_THRESH}")
```

---

## Next Steps

Future improvements to the architecture:

1. **Refactor GUI** (`Project-GUI.py` → `gui/main_window.py`)
2. **Add unit tests** (`tests/` directory)
3. **CSV reporting** (`utils/report_generator.py`)
4. **Video I/O utilities** (`utils/video_io.py`)
5. **Logging configuration** (`utils/logging_config.py`)
6. **Remove deprecated** `object_detection.py` wrapper

---

## File Size Comparison

**Before Refactoring:**
- `object_detection.py`: 943 lines (everything in one file)

**After Refactoring:**
- `models/yolov3.py`: 224 lines
- `models/weight_loader.py`: 118 lines
- `detection/detector.py`: 180 lines
- `detection/preprocessing.py`: 213 lines
- `detection/violation_checker.py`: 229 lines
- `object_detection.py`: 58 lines (backward compatibility only)

**Total:** Cleaner organization with clear separation of concerns!

---

## Support

For questions about the architecture or migration:
1. Review this document and module docstrings
2. Check function examples in docstrings
3. See `main.py` for complete usage example
4. Report issues at: https://github.com/anthropics/claude-code/issues
