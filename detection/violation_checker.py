"""
Traffic Violation Detection and Bounding Box Utilities

This module contains the BoundBox class for representing detections and
the draw_boxes function for annotating images with violation detection.
"""

import numpy as np
import cv2
import logging
import config
from utils.geometry import bounding_box_intersects_line


def _interval_overlap(interval_a, interval_b):
    """
    Calculate the overlap between two 1D intervals.

    Used as a helper function for computing Intersection over Union (IoU)
    between bounding boxes.

    Args:
        interval_a (tuple): First interval as (start, end)
        interval_b (tuple): Second interval as (start, end)

    Returns:
        float: Length of overlap between intervals. Returns 0 if no overlap.

    Example:
        >>> _interval_overlap((0, 10), (5, 15))
        5
        >>> _interval_overlap((0, 5), (10, 15))
        0
    """
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3


class BoundBox:
    """
    Represents a bounding box detection with class probabilities.

    Stores bounding box coordinates along with objectness score and
    class probabilities for all COCO classes. Provides cached access
    to predicted label and confidence score.

    Attributes:
        xmin (float): Left coordinate (normalized 0-1 or pixel coordinates)
        ymin (float): Top coordinate
        xmax (float): Right coordinate
        ymax (float): Bottom coordinate
        objness (float): Objectness score (probability of containing an object)
        classes (np.ndarray): Array of class probabilities (80 classes for COCO)
        label (int): Cached predicted class index (-1 until computed)
        score (float): Cached confidence score (-1 until computed)

    Example:
        >>> box = BoundBox(0.1, 0.2, 0.5, 0.8, 0.95, class_probs)
        >>> print(f"Class: {box.get_label()}, Confidence: {box.get_score():.2f}")
        >>> print(f"Box: ({box.xmin}, {box.ymin}, {box.xmax}, {box.ymax})")
    """
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        """
        Initialize bounding box with coordinates and predictions.

        Args:
            xmin (float): Left edge of box
            ymin (float): Top edge of box
            xmax (float): Right edge of box
            ymax (float): Bottom edge of box
            objness (float, optional): Objectness score
            classes (np.ndarray, optional): Class probability array
        """
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        """
        Get predicted class label (index with highest probability).

        Returns:
            int: Class index (0-79 for COCO dataset)

        Note:
            Result is cached for efficiency.
        """
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        """
        Get confidence score for predicted class.

        Returns:
            float: Confidence score (0-1)

        Note:
            Combines objectness and class probability.
            Result is cached for efficiency.
        """
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def bbox_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    IoU is a measure of overlap between two bounding boxes, commonly used
    in object detection for Non-Maximum Suppression (NMS).

    Args:
        box1 (BoundBox): First bounding box
        box2 (BoundBox): Second bounding box

    Returns:
        float: IoU score between 0 and 1.
            - 0 means no overlap
            - 1 means perfect overlap

    Formula:
        IoU = Area of Intersection / Area of Union

    Example:
        >>> box1 = BoundBox(0, 0, 10, 10)
        >>> box2 = BoundBox(5, 5, 15, 15)
        >>> iou = bbox_iou(box1, box2)
        >>> print(f"IoU: {iou:.2f}")
    """
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union


def draw_boxes(image, boxes, line, labels, obj_thresh, dcnt):
    """
    Draw bounding boxes on image and detect traffic violations.

    Draws detection boxes on the image with different colors:
    - Green: Compliant vehicles (not crossing line)
    - Red: Violation detected (crossing traffic signal line)

    Also draws the traffic signal line and saves violation snapshots.

    Args:
        image (np.ndarray): Input image (will be modified in-place)
        boxes (list[BoundBox]): List of detected bounding boxes
        line (list): Traffic line coordinates [(x1, y1), (x2, y2)]
        labels (list[str]): Class label names (COCO labels)
        obj_thresh (float): Confidence threshold for displaying boxes
        dcnt (int): Frame counter for naming violation snapshots

    Returns:
        np.ndarray: Annotated image with boxes and line drawn

    Side Effects:
        - Saves violation snapshots to config.DETECTED_IMAGES_DIR
        - Displays violation window with cv2.imshow (if not headless)

    Violation Detection:
        A violation is detected when any edge of a vehicle's bounding box
        intersects with the traffic signal line, indicating the vehicle
        has crossed the red signal.

    Example:
        >>> frame = cv2.imread('traffic.jpg')
        >>> boxes = [...]  # After detection and NMS
        >>> line = [(100, 200), (500, 200)]  # Horizontal line
        >>> annotated = draw_boxes(frame, boxes, line, LABELS, 0.5, 1)
        >>> cv2.imwrite('annotated.jpg', annotated)
    """
    logging.debug(f"Drawing boxes with line: {line}")

    for box in boxes:
        label_str = ''
        label = -1

        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                label_str += labels[i]
                label = i
                logging.debug(f"{labels[i]}: {box.classes[i]*100:.2f}%")
                logging.debug(f"Line: ({line[0][0]}, {line[0][1]}) ({line[1][0]}, {line[1][1]})")
                logging.debug(f"Box: ({box.xmin}, {box.ymin}) ({box.xmax}, {box.ymax})")

        if label >= 0:
            # Check if bounding box intersects with traffic signal line
            bbox_min = (box.xmin, box.ymin)
            bbox_max = (box.xmax, box.ymax)
            is_violation = bounding_box_intersects_line(bbox_min, bbox_max, line[0], line[1])

            logging.debug(f"Violation detected: {is_violation}")

            cv2.line(image, line[0], line[1], (255, 0, 0), 3)

            if is_violation:
                # Red box for violation
                cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (255,0,0), 3)
                cimg = image[box.ymin:box.ymax, box.xmin:box.xmax]
                cv2.imshow("violation", cimg)
                cv2.waitKey(5)
                violation_path = config.DETECTED_IMAGES_DIR / f"violation_{dcnt}.jpg"
                cv2.imwrite(str(violation_path), cimg)
                dcnt = dcnt+1
            else:
                # Green box for compliant vehicle
                cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)

            cv2.putText(image,
                        label_str + ' ' + str(round(box.get_score(), 2)),
                        (box.xmin, box.ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image.shape[0],
                        (0,255,0), 2)

    return image
