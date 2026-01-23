import numpy as np
import cv2 as cv
import os
from detector import runDetection

# load the video file
video = cv.VideoCapture('videoTest.mp4')

while video.isOpened():
    ret, frame = video.read()

    # if frame not found
    if not ret:
        print("Can't get frame. Exiting")
        break

    # convert the current frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # the frame to process
    cv.imwrite("currentFrame.jpg", frame)

    # return bounding boxes
    points = runDetection("currentFrame.jpg")
    boxes = points[0].boxes.xyxy.numpy()
    classes = points[0].boxes.cls.numpy().astype(int)
    confs = points[0].boxes.conf.numpy()
    names = points[0].names

    # add boxes to frame, classes, confidence
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.putText(frame, f"{names[classes[i]]} {confs[i]:.2f}", (x1, y1 - 5),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # show frame
    cv.imshow('frame', frame)
    # currently wait 1 ms for each frame (change if needed) (press q to quit)
    if cv.waitKey(1) == ord('q'):
        break

video.release()
cv.destroyAllWindows()