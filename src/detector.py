from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')

def runDetection(imageName):
	detecton = model.predict(source=imageName, conf=0.25, save=False)
	return detecton
