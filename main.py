import os
import cv2 as cv
import time
import threading
import base64

from src.detector import Detector
from src.video_handler import VideoHandler
from src.utils import draw_detections
from src.point_cloud import PointCloudGenerator as pcg

from nicegui import app, ui

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

LOCAL_VIDEO_PATH = os.path.join(PROJECT_ROOT, 'data', 'input_videos', 'videoTest.mp4')
RTSP_URL = "rtsp://192.168.2.23:8554/stream"
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'output_videos', 'output.mp4')
RGBD_VIDEO_DIR = os.path.join(
    PROJECT_ROOT, 'data', 'input_RGBD_videos', 'rgbd_dataset_freiburg1_room'
)

SOURCE_MAP = {
    'Local Video File': LOCAL_VIDEO_PATH,
    'Laptop Webcam': 0,
    'Phone Camera (RTSP)': RTSP_URL,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def frame_to_data_url(frame: cv.typing.MatLike) -> str:
    """encode an OpenCV frame as a JPEG data URL for UI display."""
    _, buf = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@ui.page('/')
def main_page():
    state = {'running': False}

    ui.label('COMP 4990 — Computer Vision').classes('text-2xl font-bold mb-4')

    with ui.row().classes('w-full gap-6 items-start'):
        # left side
        with ui.column().classes('flex-1 min-w-0'):
            video_image = ui.interactive_image('').classes('w-full rounded shadow bg-gray-900')
            status = ui.label('Status: Ready').classes('text-green-600 font-semibold mt-1')

        # right side
        with ui.column().classes('w-64 gap-2'):
            source_select = ui.select(
                list(SOURCE_MAP.keys()),
                value='Local Video File',
                label='Video Source',
            ).classes('w-full')

            def run_detection():
                if state['running']:
                    log.push('Detection is already running.')
                    return
                source = SOURCE_MAP[source_select.value]
                status.set_text('Status: Running...')
                status.classes(remove='text-green-600 text-red-600').classes('text-yellow-600')
                log.push(f'Starting: {source_select.value}')

                def detection_loop():
                    state['running'] = True
                    def safe_log(msg: str) -> None:
                        try:
                            log.push(msg)
                        except Exception:
                            print(f'[log fallback] {msg}')

                    try:
                        detector = Detector()
                        handler = VideoHandler(source, OUTPUT_PATH)
                        safe_log(
                            f'Connected: {handler.width}x{handler.height} @ {handler.fps} FPS'
                        )

                        # --- Optimisation constants ---
                        # Run YOLO only once every N frames; reuse cached result in between.
                        DETECT_EVERY_N = 3
                        # Cap UI repaints at this many frames per second to avoid
                        # saturating the main thread with JPEG-encode + base64 work.
                        UI_MAX_FPS = 20
                        ui_interval = 1.0 / UI_MAX_FPS

                        frame_count = 0
                        last_results = None
                        last_ui_update = 0.0

                        while state['running']:
                            ret, frame = handler.get_frame()
                            if not ret:
                                if handler.is_live:
                                    time.sleep(0.01)
                                    continue
                                else:
                                    safe_log('Video finished.')
                                    break
                            if frame is None:
                                continue

                            PROCESS_WIDTH = 640
                            scale = PROCESS_WIDTH / frame.shape[1]
                            pf = cv.resize(
                                frame, (PROCESS_WIDTH, int(frame.shape[0] * scale))
                            )

                            # Only run YOLO on every Nth frame; reuse last bbox on skipped frames
                            frame_count += 1
                            if last_results is None or frame_count % DETECT_EVERY_N == 0:
                                last_results = detector.detect(pf)

                            pf = draw_detections(pf, last_results)

                            out_frame = cv.resize(pf, (handler.width, handler.height))
                            handler.write_frame(out_frame)

                            # Throttle UI updates — only push a new image when enough
                            # time has elapsed since the last repaint.
                            now = time.monotonic()
                            if now - last_ui_update >= ui_interval:
                                try:
                                    video_image.set_source(frame_to_data_url(pf))
                                except Exception:
                                    pass
                                last_ui_update = now

                        handler.release()
                        safe_log(f'Saved to: {OUTPUT_PATH}')
                    except Exception as e:
                        safe_log(f'Error: {e}')
                    finally:
                        # Always reset running state — even if log/status UI calls fail.
                        state['running'] = False
                        try:
                            status.set_text('Status: Ready')
                            status.classes(remove='text-yellow-600 text-red-600').classes(
                                'text-green-600'
                            )
                        except Exception:
                            pass

                threading.Thread(target=detection_loop, daemon=True).start()

            def stop_detection():
                if state['running']:
                    state['running'] = False
                    log.push('Stopping...')
                else:
                    log.push('Nothing is running.')

            def run_point_cloud():
                log.push('Generating point cloud...')
                status.set_text('Status: Generating point cloud...')
                status.classes(remove='text-green-600 text-red-600').classes('text-yellow-600')
                pc_progress.set_value(0)

                def on_progress(processed, total_samples):
                    pc_progress.set_visibility(True)
                    pc_progress_label.set_visibility(True)
                    pc_progress.set_value(f'{(processed / total_samples) * 100:.1f}%')
                    pc_progress_label.set_text(f'Frame {processed} / {total_samples}')

                def generate_point_cloud():
                    try:
                        pcg.generate_from_video(RGBD_VIDEO_DIR, on_progress=on_progress)
                        log.push('Point cloud generation complete!')
                    except Exception as e:
                        log.push(f'Error: {e}')
                    finally:
                        status.set_text('Status: Ready')
                        status.classes(remove='text-yellow-600').classes('text-green-600')

                threading.Thread(target=generate_point_cloud, daemon=True).start()

            ui.button('Run Detection', on_click=run_detection).classes(
                'w-full bg-blue-600 text-white'
            )
            ui.button('Stop', on_click=stop_detection).classes(
                'w-full bg-red-600 text-white'
            )
            ui.button('Generate Point Cloud', on_click=run_point_cloud).classes(
                'w-full bg-green-700 text-white'
            )

            pc_progress_label = ui.label('Frame Generation Progress...').classes('text-sm text-gray-500 mt-2')
            pc_progress_label.set_visibility(False)
            pc_progress = ui.linear_progress(0).classes('w-full')
            pc_progress.set_visibility(False)

    log = ui.log(max_lines=30).classes('w-full h-40 mt-4 font-mono text-sm')


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
app.native.window_args['resizable'] = True
app.native.settings['ALLOW_DOWNLOADS'] = True

ui.run(native=True, window_size=(1100, 750), title='COMP 4990 — Computer Vision')
