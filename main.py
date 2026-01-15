import cv2
import numpy as np
from ultralytics import YOLO

def test_opencv():
    """Test if OpenCV is installed correctly"""
    print(f"OpenCV version: {cv2.__version__}")
    print("✓ OpenCV installed successfully")

def test_yolo():
    """Test if YOLO is installed correctly"""
    try:
        # This will download the model first time (takes a minute)
        model = YOLO('yolov8n.pt')  # nano version (smallest)
        print("✓ YOLO installed successfully")
        return model
    except Exception as e:
        print(f"✗ YOLO installation failed: {e}")
        return None

def test_basic_detection():
    """Test YOLO on a simple test image"""
    model = YOLO('yolov8n.pt')
    
    # Create a simple test image (blue square)
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    test_img[200:400, 200:400] = [255, 0, 0]  # Blue square
    
    # Run detection
    results = model(test_img)
    print("✓ Basic detection test completed")
    print(f"  Detected {len(results[0].boxes)} objects")

if __name__ == "__main__":
    print("=" * 50)
    print("Environment Setup Verification")
    print("=" * 50)
    
    test_opencv()
    test_yolo()
    test_basic_detection()
    
    print("\n✓ All tests passed! Environment is ready.")