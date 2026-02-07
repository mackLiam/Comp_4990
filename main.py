import os
import cv2 as cv
import numpy as np
import open3d as o3d
from src.detector import Detector
from src.video_handler import VideoHandler
from src.utils import draw_detections


def run_object_detection():
    """Run the YOLO object detection on video source."""
    # Configuration
    project_root = os.path.dirname(os.path.abspath(__file__))
    local_video = os.path.join(project_root, 'data', 'input_videos', 'videoTest.mp4')
    rtsp_url = "rtsp://192.168.2.17:8554/stream"
    output_path = os.path.join(project_root, 'data', 'output_videos', 'output.mp4')
    
    # Terminal Menu
    print("\n" + "="*30)
    print(" YOLO VIDEO SOURCE SELECTOR ")
    print("="*30)
    print("1. Local Video File (videoTest.mp4)")
    print("2. Laptop Webcam")
    print("3. Phone Camera (RTSP)")
    print("="*30)
    
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

def load_intrinsics(intrinsics_path):
    """Load camera intrinsic parameters from a text file.

    The file contains a 3x3 camera matrix in row-major order as space-separated values:
    [fx 0 cx]
    [0 fy cy]
    [0  0  1]

    Args:
        intrinsics_path: Path to the intrinsics file

    Returns:
        dict: Dictionary containing the intrinsic parameters (fx, fy, cx, cy)
    """
    try:
        with open(intrinsics_path, 'r') as f:
            values = list(map(float, f.readline().strip().split()))
            
            # Extract the camera matrix values
            # The values are in order: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            intrinsics = {
                'fx': values[0],  # fx
                'fy': values[4],  # fy
                'cx': values[2],  # cx
                'cy': values[5]   # cy
            }
            
            return intrinsics
    except Exception as e:
        print(f"Error loading intrinsics from {intrinsics_path}: {e}")
        print("Using default Kinect v1 intrinsics")
        return {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}

def generate_points():
    print("\n" + "="*30)
    print(" POINT CLOUD GENERATION")
    print("="*30)

    # Load camera intrinsics
    intrinsics = load_intrinsics('./data/input_RGBD_images/NYU0001/intrinsics.txt')
    fx = intrinsics['fx']
    fy = intrinsics['fy']
    cx = intrinsics['cx']
    cy = intrinsics['cy']

    print(f"Using camera intrinsics:")
    print(f"  fx: {fx}, fy: {fy}")
    print(f"  cx: {cx}, cy: {cy}")
    
    # Load RGB and depth images
    rgb_source = "./data/input_RGBD_images/NYU0001/NYU0001.jpg"
    depth_source = "./data/input_RGBD_images/NYU0001/NYU0001.png"
    rgb = cv.imread(rgb_source)
    depth = cv.imread(depth_source, cv.IMREAD_UNCHANGED)

    # Changing depth metric to meters
    depth_m = depth.astype('float32') / 1000.0

    if rgb is None or depth is None:
        print("Error: Could not load input images")
        return

    print("\nSuccessfully loaded RGB and depth images.")
    print(f"RGB shape: {rgb.shape}, Depth shape: {depth.shape}")

    print("\nBeginning point generation...")

    # Creating pixel coordinate grid
    height, width = depth_m.shape[:2]

    u_coords, v_coords = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )

    u = u_coords.flatten()
    v = v_coords.flatten()
    z = depth_m.flatten()

    # Filtering invalid depth values
    valid = z > 0
    u = u[valid]
    v = v[valid]
    z = z[valid]

    # Projecting pixels to 3D points
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Creating point cloud
    points = np.vstack((x, y, z)).T

    rgb_flat = rgb.reshape(-1, 3)
    colors = rgb_flat[valid] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Flip orientation upright
    pcd.transform([
        [1, 0, 0, 0],
        [0,-1, 0, 0],
        [0, 0,-1, 0],
        [0, 0, 0, 1]
    ])

    o3d.visualization.draw_geometries([pcd])

def main():
    """Main CLI entry point that allows selection between different functionalities."""
    while True:
        print("\n" + "="*30)
        print(" MAIN MENU ")
        print("="*30)
        print("1. Run Object Detection")
        print("2. Generate Points")
        print("3. Exit")
        print("="*30)
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            run_object_detection()
        elif choice == '2':
            generate_points()
        elif choice == '3':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()
