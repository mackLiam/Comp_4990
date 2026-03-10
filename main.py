import os
import cv2 as cv
import time
from src.detector import Detector
from src.video_handler import VideoHandler
from src.utils import draw_detections
from src.point_cloud import PointCloudGenerator as pcg

# Constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LOCAL_VIDEO_PATH = os.path.join(PROJECT_ROOT, 'data', 'input_videos', 'videoTest.mp4')
RTSP_URL = "rtsp://10.72.87.216:8554/stream"
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'output_videos', 'output.mp4')

RGBD_VIDEO_DIR = os.path.join(
    PROJECT_ROOT,
    'data',
    'input_RGBD_videos',
    'rgbd_dataset_freiburg1_room'
)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def show_menu_header(title):
    """Display a standardized menu header."""
    clear_screen()
    print("\n" + "="*40)
    print(f" {title} ")
    print("="*40)

def run_object_detection():
    """Run the YOLO object detection on video source."""
    show_menu_header("YOLO VIDEO SOURCE SELECTOR")

    # Configuration
    local_video = LOCAL_VIDEO_PATH
    rtsp_url = RTSP_URL
    output_path = OUTPUT_PATH

    print("1. Local Video File (videoTest.mp4)")
    print("2. Laptop Webcam")
    print("3. Phone Camera (RTSP)")
    print("4. Back to Main Menu")
    print("="*40)

    choice = input("\nSelect an option (1-4): ").strip()

    if choice == '1':
        source = local_video
    elif choice == '2':
        source = 0
    elif choice == '3':
        source = rtsp_url
    elif choice == '4':
        return
    else:
        print("Invalid choice. Returning to main menu...")
        time.sleep(1)
        return

    # Initialize Modules
    try:
        print(f"\nInitializing detector...")
        detector = Detector()
        print(f"Connecting to source: {source}...")
        handler = VideoHandler(source, output_path)
        print(f"Successfully connected to {source} at {handler.width}x{handler.height} @ {handler.fps} FPS")
    except Exception as e:
        print(f"Error during initialization: {e}")
        time.sleep(2)
        return

    # Set up the window
    window_name = 'YOLO Object Detection'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    # Main Loop
    prev_time = time.time()
    print("\nProcessing started... Press 'q' or close window to quit.")
    
    while True:
        ret, frame = handler.get_frame()
        if not ret:
            if handler.is_live:
                time.sleep(0.01)
                continue
            else:
                print("Finished or lost connection to source.")
                break

        if frame is None:
            continue

        # Resize for faster processing if frame is very large
        # We work on a copy/resized version for display and detection
        # but keep the original for high-quality saving if needed.
        # However, for 'real-time' feel, let's process at 640px width.
        PROCESS_WIDTH = 640
        scale = PROCESS_WIDTH / frame.shape[1]
        process_frame = cv.resize(frame, (PROCESS_WIDTH, int(frame.shape[0] * scale)))

        # Detection -> Drawing (on the resized 'process_frame')
        results = detector.detect(process_frame)
        processed_frame = draw_detections(process_frame, results)

        # Save the result (we save the smaller processed frame to keep writer fast)
        # Note: handler.out was initialized with original resolution. 
        # For simplicity, if we are in live mode, we might want to skip writing 
        # or resize back. Let's resize back for the output file to keep it consistent.
        out_frame = cv.resize(processed_frame, (handler.width, handler.height))
        handler.write_frame(out_frame)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time
        cv.putText(processed_frame, f"FPS: {fps:.1f}", (20, 40), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the processed frame (already resized to PROCESS_WIDTH)
        cv.imshow(window_name, processed_frame)

        # Check for exit conditions
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    # Cleanup
    handler.release()
    cv.destroyAllWindows()
    print(f"\nProcessing complete! Results saved to: {output_path}")

def generate_from_video_menu():
    """Menu for video-based point cloud generation options."""
    while True:
        show_menu_header("VIDEO GENERATION OPTIONS")
        print("1. Watch Video (Normal Playback)")
        print("2. Generate Point Cloud from Video")
        print("3. Back to Points Menu")
        print("="*40)

        choice = input("\nSelect an option (1-3): ").strip()

        if choice == '1':
            VideoHandler.play_from_images(os.path.join(RGBD_VIDEO_DIR, 'rgb'))
        elif choice == '2':
            pcg.generate_from_video(RGBD_VIDEO_DIR)
        elif choice == '3':
            break
        else:
            print("\nInvalid choice. Please try again.")

def generate_points_menu():
    """Menu for point cloud generation options."""
    while True:
        show_menu_header("POINT CLOUD GENERATION")
        print("1. Generate from Video")
        print("2. Back to Main Menu")
        print("="*40)

        choice = input("\nSelect an option (1-2): ").strip()

        if choice == '1':
            generate_from_video_menu()
        elif choice == '2':
            break
        else:
            print("\nInvalid choice. Please try again.")

def main():
    """Main CLI entry point that allows selection between different functionalities."""
    while True:
        show_menu_header("MAIN MENU")
        print("1. Run Object Detection")
        print("2. Generate Points")
        print("3. Exit")
        print("="*40)
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            run_object_detection()
        elif choice == '2':
            generate_points_menu()
        elif choice == '3':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
