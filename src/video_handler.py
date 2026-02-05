import cv2 as cv
import os

class VideoHandler:
    def __init__(self, input_path, output_path):
        """Initializes video input and output streams."""
        self.cap = cv.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))

        # Ensure output folder exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Setup VideoWriter
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.out = cv.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

    def get_frame(self):
        """Reads the next frame from source."""
        return self.cap.read()

    def write_frame(self, frame):
        """Writes frame to the output file."""
        self.out.write(frame)

    def release(self):
        """Release resources and close windows."""
        self.cap.release()
        self.out.release()
        cv.destroyAllWindows()