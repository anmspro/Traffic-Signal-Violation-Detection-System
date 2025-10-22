"""
DEPRECATED: Legacy object_detection module

This module is deprecated and maintained only for backward compatibility.
Please update your imports to use the new modular structure:

OLD:
    from object_detection import detector, preprocess_input, draw_boxes

NEW:
    from detection import detector, preprocess_input, draw_boxes
    from models import make_yolov3_model, WeightReader

This file will be removed in a future version.
"""

import warnings

# Show deprecation warning
warnings.warn(
    "object_detection module is deprecated. "
    "Use 'from detection import ...' and 'from models import ...' instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from new modular structure for backward compatibility
from models import make_yolov3_model, WeightReader
from detection import (
    ViolationDetector,
    detector,
    preprocess_input,
    decode_netout,
    correct_yolo_boxes,
    do_nms,
    BoundBox,
    bbox_iou,
    draw_boxes
)

__all__ = [
    # Model components
    'make_yolov3_model',
    'WeightReader',
    # Detection components
    'ViolationDetector',
    'detector',
    'preprocess_input',
    'decode_netout',
    'correct_yolo_boxes',
    'do_nms',
    # Violation checking
    'BoundBox',
    'bbox_iou',
    'draw_boxes',
]
