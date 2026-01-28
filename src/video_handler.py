import numpy as np
import cv2 as cv
import os
from detector import runDetection

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the project root (one level up from src)
project_root = os.path.dirname(script_dir)
# Define paths relative to the project root
video_path = os.path.join(project_root, 'data', 'input_videos', 'videoTest.mp4')
current_frame_path = os.path.join(project_root, 'data', 'input_videos', 'currentFrame.jpg')
output_dir = os.path.join(project_root, 'data', 'output_videos')
output_path = os.path.join(output_dir, 'output.mp4')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# load the video file
video = cv.VideoCapture(video_path)

# Get video properties for the writer
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv.CAP_PROP_FPS))

# Initialize VideoWriter
# 'mp4v' is a common codec for mp4 files
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while video.isOpened():
    ret, frame = video.read()

    # if frame not found
    if not ret:
        print("Can't get frame. Exiting")
        break

    # convert the current frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # the frame to process
    cv.imwrite(current_frame_path, frame)

    # return bounding boxes
    points = runDetection(current_frame_path)
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
    
    # write the frame to the output video
    out.write(frame)

    # currently wait 1 ms for each frame (change if needed) (press q to quit)
    if cv.waitKey(1) == ord('q'):
        break

video.release()
out.release()
cv.destroyAllWindows()