"""
Image Preprocessing and Output Decoding for YOLOv3

This module handles all preprocessing operations for YOLOv3 detection including
letterboxing, normalization, output decoding, box correction, and NMS.
"""

import numpy as np
import cv2
from .violation_checker import BoundBox, bbox_iou


def _sigmoid(x):
    """
    Apply sigmoid activation function.

    Args:
        x (np.ndarray or float): Input value(s)

    Returns:
        np.ndarray or float: Sigmoid activation output in range (0, 1)

    Example:
        >>> _sigmoid(0)
        0.5
        >>> _sigmoid(np.array([0, 1, -1]))
        array([0.5, 0.73105858, 0.26894142])
    """
    return 1. / (1. + np.exp(-x))


def preprocess_input(image, net_h, net_w):
    """
    Preprocess input image for YOLOv3 detection.

    Resizes image to network input size while maintaining aspect ratio using
    letterboxing (padding with gray). Converts BGR to RGB and normalizes
    pixel values to [0, 1].

    Args:
        image (np.ndarray): Input image in BGR format (H, W, 3)
        net_h (int): Network input height (typically 416)
        net_w (int): Network input width (typically 416)

    Returns:
        np.ndarray: Preprocessed image with shape (1, net_h, net_w, 3)
            - Normalized to [0, 1]
            - RGB format
            - Letterboxed to maintain aspect ratio
            - Batch dimension added

    Example:
        >>> frame = cv2.imread('image.jpg')  # (720, 1280, 3)
        >>> preprocessed = preprocess_input(frame, 416, 416)
        >>> print(preprocessed.shape)
        (1, 416, 416, 3)
    """
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)/new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)/new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image


def decode_netout(netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
    """
    Decode YOLOv3 network output into bounding boxes.

    Converts raw network predictions into interpretable bounding boxes with
    class probabilities. Applies objectness threshold to filter low-confidence
    detections.

    Args:
        netout (np.ndarray): Network output tensor (grid_h, grid_w, 255)
        anchors (list): Anchor box dimensions for this scale [w1,h1, w2,h2, w3,h3]
        obj_thresh (float): Objectness threshold (0-1) for filtering detections
        nms_thresh (float): NMS threshold (not used in this function)
        net_h (int): Network input height
        net_w (int): Network input width

    Returns:
        list[BoundBox]: List of detected bounding boxes with class probabilities

    Note:
        Each grid cell predicts 3 bounding boxes (one per anchor).
        Output shape 255 = 3 * (5 + 80) where:
            - 5 = (x, y, w, h, objectness)
            - 80 = class probabilities for COCO dataset

    Example:
        >>> yolos = model.predict(preprocessed_image)
        >>> boxes = decode_netout(yolos[0][0], anchors[0], 0.5, 0.45, 416, 416)
        >>> print(f"Detected {len(boxes)} boxes")
    """
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w  # Integer division
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            #objectness = netout[..., :4]

            if(objectness.all() <= obj_thresh): continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height

            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]

            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            #box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, None, classes)

            boxes.append(box)

    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    """
    Correct bounding box coordinates to match original image dimensions.

    YOLOv3 predictions are relative to the letterboxed image. This function
    transforms coordinates back to the original image space, accounting for
    the letterboxing (padding) applied during preprocessing.

    Args:
        boxes (list[BoundBox]): List of bounding boxes to correct (modified in-place)
        image_h (int): Original image height
        image_w (int): Original image width
        net_h (int): Network input height (e.g., 416)
        net_w (int): Network input width (e.g., 416)

    Returns:
        None: Modifies boxes in-place

    Note:
        This function accounts for the aspect-ratio-preserving resize and
        letterboxing applied in preprocess_input().

    Example:
        >>> boxes = decode_netout(...)  # boxes in network coordinates
        >>> correct_yolo_boxes(boxes, 720, 1280, 416, 416)
        >>> # boxes now in original image coordinates (0-720, 0-1280)
    """
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h

    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def do_nms(boxes, nms_thresh):
    """
    Apply Non-Maximum Suppression (NMS) to remove duplicate detections.

    NMS suppresses overlapping bounding boxes, keeping only the highest
    confidence detection for each object. Applied per-class to handle
    multi-class detection scenarios.

    Args:
        boxes (list[BoundBox]): List of bounding boxes (modified in-place)
        nms_thresh (float): IoU threshold (0-1) for suppression.
            - Higher values = more aggressive suppression
            - Typical value: 0.45

    Returns:
        None: Modifies boxes in-place by zeroing out class probabilities
              of suppressed boxes

    Algorithm:
        For each class:
        1. Sort boxes by class probability (descending)
        2. For each box, suppress all lower-scoring boxes with IoU > threshold
        3. Suppressed boxes have their class probability set to 0

    Example:
        >>> boxes = decode_netout(...)
        >>> print(f"Before NMS: {len(boxes)} boxes")
        >>> do_nms(boxes, 0.45)
        >>> # Filter out suppressed boxes
        >>> boxes = [box for box in boxes if box.get_score() > 0]
        >>> print(f"After NMS: {len(boxes)} boxes")
    """
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
