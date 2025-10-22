"""
YOLOv3 Detection Pipeline for Traffic Violation Detection

This package contains the complete detection pipeline including preprocessing,
object detection, violation checking, and the lazy-loading detector wrapper.
"""

from .detector import ViolationDetector, detector
from .preprocessing import preprocess_input, decode_netout, correct_yolo_boxes, do_nms
from .violation_checker import BoundBox, bbox_iou, draw_boxes

__all__ = [
    'ViolationDetector',
    'detector',
    'preprocess_input',
    'decode_netout',
    'correct_yolo_boxes',
    'do_nms',
    'BoundBox',
    'bbox_iou',
    'draw_boxes',
]
