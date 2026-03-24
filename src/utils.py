import cv2 as cv

"""
This module handles drawing bounding boxes and labels on the frame.
"""

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
        
        MINT = (179, 253, 105)  # #69fdb3 in BGR
        cv.rectangle(frame, (x1, y1), (x2, y2), MINT, 2)
        
        # Draw label background and text
        label = f"{names[classes[i]]} {confs[i]:.2f}"
        (w, h), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), MINT, -1)
        cv.putText(frame, label, (x1, y1 - 5), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
    return frame
