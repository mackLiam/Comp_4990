import cv2 as cv
import os
import sys
import time

"""
This module handles video input and output.
"""


class VideoHandler:
    def __init__(self, source, output_path):
        """
        Initializes video input and output streams.
        """
        self.cap = cv.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        # Get video properties
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))

        # Fallback for cameras that don't report FPS correctly
        if self.fps <= 0:
            self.fps = 30

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Setup VideoWriter
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.out = cv.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

    def get_frame(self):
        """Reads the next frame from the source."""
        return self.cap.read()

    def write_frame(self, frame):
        """Writes the processed frame to the output file."""
        self.out.write(frame)

    def release(self):
        """Releases all resources and closes windows."""
        self.cap.release()
        self.out.release()
        cv.destroyAllWindows()

    @staticmethod
    def play_from_images(image_dir):
        """Load and play video from image files in a directory.

        Args:
            image_dir: Path to the directory containing image files
        """
        if not os.path.exists(image_dir):
            print(f"Error: Image directory not found at {image_dir}")
            return

        # Get and sort jpg files numerically
        image_files = [
            f for f in os.listdir(image_dir) 
            if f.endswith('.jpg')
        ]
        image_files.sort(key=lambda x: int(x.split('-')[0].lstrip('0')))

        if not image_files:
            print("Error: No jpg files found in the image directory")
            return

        print(f"Found {len(image_files)} image files")
        print("Press 'q' or close window to quit")
        print("\nStarting playback...")

        # Create window with native size
        window_name = 'Video from Images'
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        # Playback loop
        for i, image_file in enumerate(image_files):
            frame = cv.imread(os.path.join(image_dir, image_file))
            if frame is None:
                print(f"Warning: Could not read image {image_file}")
                continue

            # Display frame at native size
            cv.imshow(window_name, frame)

            # Progress update
            sys.stdout.write(f"\rFrame {i+1}/{len(image_files)}: {image_file}".ljust(60))
            sys.stdout.flush()

            # Exit on 'q' or window close
            if cv.waitKey(30) & 0xFF == ord('q') or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
                break

        # Cleanup
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()
        cv.destroyAllWindows()
        print("\nVideo playback complete!")
        time.sleep(1)
