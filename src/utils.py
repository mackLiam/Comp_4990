import cv2 as cv
import numpy as np
from collections import Counter

"""
This module handles drawing bounding boxes and labels on the frame.
"""
MINT = (179, 253, 105)  # #69fdb3 in BGR


def draw_detections(frame, detections):
    """
    Draws bounding boxes and labels on the frame.
    """
    boxes = detections.boxes.xyxy.numpy()
    classes = detections.boxes.cls.numpy().astype(int)
    confs = detections.boxes.conf.numpy()
    names = detections.names

    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        

        cv.rectangle(frame, (x1, y1), (x2, y2), MINT, 2)
        
        # Draw label background and text
        label = f"{names[classes[i]]} {confs[i]:.2f}"
        (w, h), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), MINT, -1)
        cv.putText(frame, label, (x1, y1 - 5), 
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return frame

def draw_tracks(frame, result, trails):

    if result.boxes.id is None:
        return frame

    ids    = result.boxes.id.numpy().astype(int)
    boxes  = result.boxes.xyxy.numpy()
    confs  = result.boxes.conf.numpy()
    classes = result.boxes.cls.numpy().astype(int)
    names  = result.names

    for i, tid in enumerate(ids):
        x1, y1, x2, y2 = map(int, boxes[i])
        cv.rectangle(frame, (x1, y1), (x2, y2), MINT, 2)

        label = f"#{tid} {names[classes[i]]} {confs[i]:.2f}"
        (w, h), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), MINT, -1)
        cv.putText(frame, label, (x1, y1 - 5),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw motion trail
        pts = trails.get(tid, [])
        for j in range(1, len(pts)):
            alpha = j / len(pts)          # fade older points
            colour = tuple(int(c * alpha) for c in MINT)
            cv.line(frame, pts[j-1], pts[j], colour, 2)

    return frame

def count_objects(result) -> dict:
    """Returns a dict mapping class name -> count from a YOLO result."""
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return {}
    classes = result.boxes.cls.numpy().astype(int)
    names = result.names
    # Build a list of class name strings, one entry per detected box
    class_name_list = [names[c] for c in classes]
    # Count how many times each name appears
    counts = Counter(class_name_list)

    return dict(counts)
