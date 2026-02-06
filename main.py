import os
import cv2 as cv
from src.detector import Detector
from src.video_handler import VideoHandler
from src.utils import draw_detections

def main():
    # 1. Configuration
    project_root = os.path.dirname(os.path.abspath(__file__))
    local_video = os.path.join(project_root, 'data', 'input_videos', 'videoTest.mp4')
    rtsp_url = "rtsp://192.168.2.17:8554/stream"
    output_path = os.path.join(project_root, 'data', 'output_videos', 'output.mp4')
    
    # 2. Terminal Menu
    print("\n" + "="*30)
    print(" YOLO VIDEO SOURCE SELECTOR ")
    print("="*30)
    print("1. Local Video File (videoTest.mp4)")
    print("2. Laptop Webcam")
    print("3. Phone Camera (RTSP)")
    
    choice = input("\nSelect an option (1-3): ").strip()

    if choice == '1':
        source = local_video
    elif choice == '2':
        source = 0
    elif choice == '3':
        source = rtsp_url
    else:
        print("Invalid choice. Exiting.")
        return

    # 3. Initialize Modules
    try:
        detector = Detector()
        handler = VideoHandler(source, output_path)
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    # Set up the window so it can be manually resized
    window_name = 'YOLO Object Detection'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    print("\nProcessing started... Press 'q' or close window to quit.")

    # 4. Main Loop
    while True:
        ret, frame = handler.get_frame()
        
        if not ret:
            print("Finished or lost connection to source.")
            break
        
        # Detection -> Drawing
        results = detector.detect(frame)
        processed_frame = draw_detections(frame, results)
        
        # Save original resolution
        handler.write_frame(processed_frame)

        # RESIZE FOR INITIAL DISPLAY (still useful for high-res inputs)
        display_frame = processed_frame
        if processed_frame.shape[0] > 720:
            scale = 720 / processed_frame.shape[0]
            new_size = (int(processed_frame.shape[1] * scale), 720)
            display_frame = cv.resize(processed_frame, new_size)

        cv.imshow(window_name, display_frame)

        # --- CLOSING LOGIC ---
        # 1. Check for 'q' key
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
        # 2. Check if the (X) was clicked (if the window property returns -1, it's closed)
        if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    # 5. Cleanup
    handler.release()
    print(f"\nProcessing complete! Results saved to: {output_path}")

if __name__ == "__main__":
    main()
