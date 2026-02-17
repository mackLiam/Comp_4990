import os
import cv2 as cv
import open3d as o3d
import time
from src.detector import Detector
from src.video_handler import VideoHandler
from src.utils import draw_detections
from src.point_cloud import PointCloudGenerator as pcg, CameraProperties

# Constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LOCAL_VIDEO_PATH = os.path.join(PROJECT_ROOT, 'data', 'input_videos', 'videoTest.mp4')
RTSP_URL = "rtsp://192.168.2.17:8554/stream"
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'output_videos', 'output.mp4')

RGB_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'data', 'input_RGBD_videos', 'brown_bm_1', 'image', '0000001-000000020431.jpg')
DEPTH_IMAGE_PATH = os.path.join(PROJECT_ROOT, 'data', 'input_RGBD_videos', 'brown_bm_1', 'depth', '0000001-000000000000.png')
INTRINSICS_PATH = os.path.join(PROJECT_ROOT, 'data', 'input_RGBD_videos', 'brown_bm_1', 'intrinsics.txt')
RGBD_VIDEO_DIR = os.path.join(PROJECT_ROOT, 'data', 'input_RGBD_videos', 'brown_bm_1')

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
        detector = Detector()
        handler = VideoHandler(source, output_path)
    except Exception as e:
        print(f"Error during initialization: {e}")
        return

    # Set up the window
    window_name = 'YOLO Object Detection'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    print("\nProcessing started... Press 'q' or close window to quit.")

    # Main Loop
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

        # Resize for display if needed
        display_frame = processed_frame
        if processed_frame.shape[0] > 720:
            scale = 720 / processed_frame.shape[0]
            new_size = (int(processed_frame.shape[1] * scale), 720)
            display_frame = cv.resize(processed_frame, new_size)

        cv.imshow(window_name, display_frame)

        # Check for exit conditions
        key = cv.waitKey(1) & 0xFF
        if key == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
            break

    # Cleanup
    handler.release()
    cv.destroyAllWindows()
    print(f"\nProcessing complete! Results saved to: {output_path}")

def generate_from_image():
    """Generate point cloud from a single RGB-D image pair."""
    show_menu_header("GENERATE FROM IMAGE")

    # Load images and intrinsics
    rgb = cv.imread(RGB_IMAGE_PATH)
    depth = cv.imread(DEPTH_IMAGE_PATH, cv.IMREAD_UNCHANGED)
    intrinsics = CameraProperties.load_intrinsics(INTRINSICS_PATH)
    
    if rgb is None or depth is None:
        print("Error: Could not load input images")
        return
    
    # Generate point cloud
    cloud = pcg.create_frame_cloud(rgb, depth, **intrinsics)
    o3d.visualization.draw_geometries([cloud])

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
            VideoHandler.play_from_images(os.path.join(RGBD_VIDEO_DIR, 'image'))
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
        print("1. Generate from Image")
        print("2. Generate from Video")
        print("3. Back to Main Menu")
        print("="*40)

        choice = input("\nSelect an option (1-3): ").strip()

        if choice == '1':
            generate_from_image()
        elif choice == '2':
            generate_from_video_menu()
        elif choice == '3':
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
