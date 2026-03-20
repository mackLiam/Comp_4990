from ultralytics import YOLO

"""
This module handles YOLO detection.
"""

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initializes the YOLO detector."""
        self.model = YOLO(model_path)

    def detect(self, frame, conf: float = 0.25):
        """
        Runs detection on a single frame (numpy array) in memory.
        :param frame: The image frame from OpenCV.
        :param conf: Confidence threshold (0.0–1.0).
        :return: Results object from Ultralytics.
        """
        # source=frame uses the array in memory instead of reading from disk
        results = self.model.predict(source=frame, conf=conf, save=False, verbose=False, imgsz=320)
        return results[0]
