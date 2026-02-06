from ultralytics import YOLO

"""
This module handles YOLO detection.
"""

class Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initializes the YOLO detector."""
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Runs detection on a single frame (numpy array) in memory.
        :param frame: The image frame from OpenCV.
        :return: Results object from Ultralytics.
        """
        # source=frame uses the array in memory instead of reading from disk
        results = self.model.predict(source=frame, conf=0.25, save=False, verbose=False)
        return results[0]
