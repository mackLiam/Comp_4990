from ultralytics import YOLO
from collections import defaultdict
import numpy as np

class Tracker:
    def __init__(self, model_path='yolov8s.pt', trail_length=50):
        self.model = YOLO(model_path)
        self.trail_length = trail_length
        # maps track_id -> list of (cx, cy) centre points
        self.trails = defaultdict(list)

    def track(self, frame, conf: float = 0.25):
        results = self.model.track(
            source=frame,
            conf=conf,
            tracker='bytetrack.yaml',   # built-in ByteTrack config
            persist=True,               # keeps track IDs consistent across calls
            save=False,
            verbose=False,
            imgsz=320,
        )
        result = results[0]

        # Update trails
        if result.boxes.id is not None:
            ids = result.boxes.id.numpy().astype(int)
            boxes = result.boxes.xyxy.numpy()
            for tid, box in zip(ids, boxes):
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                self.trails[tid].append((cx, cy))
                # keep only the last N points
                if len(self.trails[tid]) > self.trail_length:
                    self.trails[tid].pop(0)

        return result, self.trails

    def reset(self):
        self.trails.clear()