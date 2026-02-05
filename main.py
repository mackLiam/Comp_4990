import os
import cv2 as cv
from src.detector import Detector
from src.video_handler import VideoHandler
from src.utils import draw_detections

def main():
    # 1. Paths relative to this project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(project_root, 'data', 'input_videos', 'videoTest.mp4')
    output_path = os.path.join(project_root, 'data', 'output_videos', 'output.mp4')

    # 2. Init modules
    try:
        detector = Detector()
        handler = VideoHandler(video_path, output_path)
    except Exception as e:
        print(f"Error starting program: {e}")
        return

    print("Processing video (In-Memory)... Press 'q' to quit.")

    # 3. Processing Loop
    while True:
        ret, frame = handler.get_frame()
        if not ret:
            break
        
        # Detection -> Drawing -> Saving
        results = detector.detect(frame)
        processed_frame = draw_detections(frame, results)
        
        handler.write_frame(processed_frame)
        cv.imshow('Modular YOLO Detection', processed_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Cleanup
    handler.release()
    print(f"Finished! Output saved to: {output_path}")

if __name__ == "__main__":
    main()
