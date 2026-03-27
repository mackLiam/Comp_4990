import cv2 as cv
import os
import sys
import time
import threading

class VideoHandler:
    def __init__(self, source, output_path, save_video=True):
        """
        Initializes video input and output streams.
        """
        self.source = source
        self.is_live = isinstance(source, int) or (isinstance(source, str) and source.startswith("rtsp"))
        
        # Improved RTSP handling
        is_rtsp = isinstance(source, str) and source.startswith("rtsp")
        
        if is_rtsp:
            # Force TCP for RTSP streams (more stable).
            # stimeout sets the socket-level timeout in microseconds (5 000 000 µs = 5 s).
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer"
            # Use FFMPEG backend explicitly for RTSP
            self.cap = cv.VideoCapture(source, cv.CAP_FFMPEG)
            # Set buffer size to 1 to reduce latency and prevent overflow
            self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        # Get video properties (with retries for RTSP)
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv.CAP_PROP_FPS))

        # For RTSP, properties might not be available immediately
        if is_rtsp and (self.width <= 0 or self.height <= 0):
            print("Connecting to RTSP stream, waiting for properties...")
            for i in range(20):  # Try for 2 seconds
                time.sleep(0.1)
                self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                if self.width > 0 and self.height > 0:
                    break
            
            # If still not found, try reading one frame
            if self.width <= 0 or self.height <= 0:
                ret, frame = self.cap.read()
                if ret:
                    self.height, self.width = frame.shape[:2]

        if self.width <= 0 or self.height <= 0:
            self.width, self.height = 640, 480  # Default fallback
            print(f"Warning: Could not determine resolution. Using fallback: {self.width}x{self.height}")

        # Fallback for cameras that don't report FPS correctly
        if self.fps <= 0:
            self.fps = 30
            if is_rtsp:
                print(f"Warning: Could not determine FPS for RTSP. Using fallback: {self.fps}")

        # Setup VideoWriter (only if saving is requested)
        self.save_video = save_video
        if self.save_video:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            self.out = cv.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        else:
            self.out = None

        # Threaded reading for live sources
        if self.is_live:
            self.ret = False
            self.frame = None
            self.stopped = False
            self.thread = threading.Thread(target=self._update, args=())
            self.thread.daemon = True
            self.thread.start()

    def _update(self):
        """Threaded function to constantly read frames."""
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                break
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                # Small sleep to prevent tight loop on connection drop
                time.sleep(0.01)

    def get_frame(self):
        """Reads the next frame from the source. For live sources, returns the latest frame."""
        if self.is_live:
            return self.ret, self.frame
        return self.cap.read()

    def write_frame(self, frame):
        """Writes the processed frame to the output file (no-op if save_video=False)."""
        if self.save_video and self.out is not None:
            self.out.write(frame)

    def release(self):
        """Releases all resources and closes windows."""
        self.stopped = True
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()
        if self.out is not None:
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

        # Get and sort image files
        allowed_ext = ('.jpg', '.jpeg', '.png')
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(allowed_ext)
        ]

        def sort_key(name):
            stem, _ = os.path.splitext(name)
            try:
                return float(stem)
            except ValueError:
                return stem

        image_files.sort(key=sort_key)

        if not image_files:
            print("Error: No image files found in the image directory")
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
