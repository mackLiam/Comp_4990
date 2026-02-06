import cv2 as cv
import os

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