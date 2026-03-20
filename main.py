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
# UI helpers
# ---------------------------------------------------------------------------
CSS = '''<style>
  * { box-sizing: border-box; }
  body, .q-page, .nicegui-content { padding: 0 !important; margin: 0 !important; }
  .q-page { background-color: #242527 !important; }

  .app-container {
    display: flex;
    height: 100vh;
    background-color: #242527;
    overflow: hidden;
  }

  /* Sidebar */
  .sidebar {
    display: flex;
    flex-direction: column;
    background-color: #242527;
    min-width: 220px;
    padding: 10px;
    flex-shrink: 0;
  }
  .sidebar-icon {
    margin: 20px 0 0 10px;
    font-size: 36px !important;
    color: white !important;
    width: 36px;
  }
  .nav-cont {
    margin-top: 40px;
    display: flex;
    flex-direction: column;
    gap: 15px;
  }
  .nav-pill {
    border-radius: 20px;
    padding: 10px 20px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 10px;
    color: white;
    transition: background-color 0.3s, color 0.3s;
    user-select: none;
  }
  .nav-pill .q-icon, .nav-pill label { color: white !important; pointer-events: none; }
  .nav-pill.active { background-color: white; color: #242527; }
  .nav-pill.active .q-icon, .nav-pill.active label { color: #242527 !important; }
  .nav-pill:hover { background-color: white; color: #242527; }
  .nav-pill:hover .q-icon, .nav-pill:hover label { color: #242527 !important; }

  /* Main panel */
  .main-panel {
    background-color: #dedede;
    border-radius: 20px;
    flex: 1;
    margin: 5px 5px 5px 0;
    overflow: auto;
    display: flex;
    flex-direction: column;
  }
  .page-title {
    margin: 15px;
    font-size: 1.5rem;
    font-weight: bold;
    color: #111;
  }
  .top-part {
    display: flex;
    gap: 10px;
    margin-left: 10px;
    margin-top: 10px;
    width: 920px;
  }
  .video-box {
    width: 600px;
    height: 350px;
    background-color: #242527;
    border-radius: 20px;
    flex-shrink: 0;
    overflow: hidden;
    position: relative;
  }
  .video-label {
    position: absolute;
    top: 12px;
    left: 16px;
    color: rgba(255,255,255,0.55);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    pointer-events: none;
    z-index: 10;
    user-select: none;
  }

  /* Control panel */
  .control-panel {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 350px;
    width: 300px;
  }
  .run-stop { display: flex; flex-direction: column; gap: 10px; }
  .btn-run, .btn-stop {
    min-width: 300px;
    height: 70px;
    border-radius: 20px;
    font-size: 16px;
    font-weight: 600;
    color: black;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
  }
  .btn-run { background-color: #69fdb3; transition: opacity 0.5s; }
  .btn-run:hover { opacity: 0.6; }
  .btn-stop { background-color: #bcb1f3; transition: opacity 0.5s; }
  .btn-stop:hover { opacity: 0.6; }
  .btn-run label, .btn-stop label {
    pointer-events: none;
    color: white !important;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
  }

  /* Source selector */
  .source-section { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
  .source-title { font-weight: 600; color: #242527; font-size: 0.95rem; }
  .source-btn {
    width: 100%;
    padding: 10px;
    border-radius: 10px;
    text-align: left;
    background-color: #3a3b3d;
    color: #d1d5db;
    border: none;
    cursor: pointer;
    font-size: 14px;
    display: flex;
    align-items: center;
    transition: background-color 0.2s, color 0.2s;
  }
  .source-btn label { pointer-events: none; color: inherit !important; font-size: 14px; cursor: pointer; }
  .source-btn.active-src { background-color: #ffffff; color: #242527; }
  .source-btn:hover { background-color: #ffffff; color: #242527; }

  /* Bottom cards */
  .bottom-section { display: flex; gap: 10px; margin: 10px; max-width: 910px; }
  .card {
    border-radius: 15px;
    padding: 12px;
    background-color: #1e293b;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .status-card { min-width: 180px; background-color: white; }
  .progress-card { min-width: 180px; background-color: white; }
  .log-card { flex: 1; background-color: white; }
  .card-header {
    font-size: 11px;
    letter-spacing: 0.08em;
    color: black;
    opacity: 0.5;
    margin: 0;
  }
  .nicegui-log { background-color: transparent !important; }

  /* Tolerance card */
  .tolerance-slider { width: 100%; margin: 2px 0; }
  .tolerance-slider .q-slider__track { background: #69fdb3 !important; }
  .tolerance-slider .q-slider__thumb { color: #69fdb3 !important; }
  .tolerance-slider .q-slider__thumb path { stroke: #69fdb3 !important; fill: #69fdb3 !important; }
  .tolerance-slider .q-slider__track-container .q-slider__track { background: #69fdb3 !important; }
  .tolerance-slider .q-slider__selection { background: #69fdb3 !important; }
  .tolerance-input .q-field__control { background: transparent !important; border-radius: 8px; min-height: 36px; }
  .tolerance-input .q-field__native { color: #242527 !important; font-size: 14px; padding: 2px 8px; }
  .tolerance-input .q-field__bottom { display: none; }
  .tolerance-input { width: 100%; }
</style>'''


def build_sidebar(active: str) -> None:
    """Renders the shared sidebar. active = 'detection' | 'reconstruction'"""
    with ui.element('div').classes('sidebar'):
        ui.image('./ui_IMG/eye.svg').classes('sidebar-icon')
        with ui.element('div').classes('nav-cont'):
            detection_cls = 'nav-pill active' if active == 'detection' else 'nav-pill'
            recon_cls = 'nav-pill active' if active == 'reconstruction' else 'nav-pill'
            with ui.element('div').classes(detection_cls).on(
                'click', lambda: ui.navigate.to('/')
            ):
                ui.icon('search').style('font-size:16px;')
                ui.label('Detection')
            with ui.element('div').classes(recon_cls).on(
                'click', lambda: ui.navigate.to('/reconstruction')
            ):
                ui.icon('view_in_ar').style('font-size:16px;')
                ui.label('3D Reconstruction')


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@ui.page('/')
def main_page():
    state = {'running': False, 'source': 'Local Video File', 'tolerance': 0.5}

    ui.add_head_html(CSS)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def update_source_buttons(selected: str) -> None:
        for name, btn in source_btns.items():
            if name == selected:
                btn.classes(add='active-src')
            else:
                btn.classes(remove='active-src')

    def select_source(name: str) -> None:
        state['source'] = name
        update_source_buttons(name)

    def run_detection():
        if state['running']:
            log.push('Detection is already running.')
            return
        source = SOURCE_MAP[state['source']]
        status.set_text('Running...')
        status.style('color: #facc15')
        log.push(f'Starting: {state["source"]}')

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
                safe_log(f'Connected: {handler.width}x{handler.height} @ {handler.fps} FPS')

                DETECT_EVERY_N = 3
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
                    pf = cv.resize(frame, (PROCESS_WIDTH, int(frame.shape[0] * scale)))

                    frame_count += 1
                    if last_results is None or frame_count % DETECT_EVERY_N == 0:
                        last_results = detector.detect(pf, conf=state['tolerance'])

                    pf = draw_detections(pf, last_results)
                    out_frame = cv.resize(pf, (handler.width, handler.height))
                    handler.write_frame(out_frame)

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
                state['running'] = False
                try:
                    status.set_text('Ready')
                    status.style('color: #69fdb3')
                except Exception:
                    pass

        threading.Thread(target=detection_loop, daemon=True).start()

    def stop_detection():
        if state['running']:
            state['running'] = False
            log.push('Stopping...')
        else:
            log.push('Nothing is running.')

    # ── Layout ────────────────────────────────────────────────────────────────
    with ui.element('div').classes('app-container'):

        build_sidebar('detection')

        # Main panel
        with ui.element('div').classes('main-panel'):
            # Top row: video + control panel
            with ui.element('div').classes('top-part'):
                with ui.element('div').classes('video-box'):
                    ui.label('Object Detection').classes('video-label')
                    video_image = ui.interactive_image('').style(
                        'width:100%; height:100%; object-fit:contain;'
                    )

                with ui.element('div').classes('control-panel'):
                    with ui.element('div').classes('run-stop'):
                        with ui.element('button').classes('btn-run').on('click', run_detection):
                            ui.label('Run Detection')
                            ui.image('./ui_IMG/play.svg').style('width: 15px;')
                        with ui.element('button').classes('btn-stop').on('click', stop_detection):
                            ui.label('Stop')
                            ui.image('./ui_IMG/stop.svg').style('width: 15px;')

                    with ui.element('div').classes('source-section'):
                        ui.label('Video Source:').classes('source-title')
                        source_btns: dict = {}
                        for src_name in SOURCE_MAP.keys():
                            btn_el = ui.element('button').classes('source-btn').on(
                                'click', lambda n=src_name: select_source(n)
                            )
                            with btn_el:
                                ui.label(src_name)
                            source_btns[src_name] = btn_el
                        update_source_buttons(state['source'])

            # Bottom row: status + progress + log
            with ui.element('div').classes('bottom-section'):
                with ui.element('div').classes('card status-card'):
                    ui.label('Status').classes('card-header')
                    status = ui.label('Ready').style(
                        'color: #69fdb3 ; font-weight: bold; font-size: 1rem;'
                    )

                with ui.element('div').classes('card progress-card'):
                    ui.label('Tracking Tolerance').classes('card-header')
                    tolerance_slider = ui.slider(
                        min=0.0, max=1.0, step=0.01
                    ).classes('tolerance-slider').bind_value(state, 'tolerance')
                    tolerance_input = ui.number(
                        min=0.0, max=1.0, step=0.01, format='%.2f'
                    ).classes('tolerance-input').bind_value(state, 'tolerance')

                with ui.element('div').classes('card log-card'):
                    log = ui.log(max_lines=30).style(
                        'height: 120px; color: black;'
                        ' font-family: monospace; font-size: 0.875rem; background: transparent; border: none;'
                    )


# ---------------------------------------------------------------------------
# 3D Reconstruction page
# ---------------------------------------------------------------------------
@ui.page('/reconstruction')
def reconstruction_page():
    state = {'running': False, 'previewing': False}

    ui.add_head_html(CSS)

    def run_point_cloud():
        if state['running']:
            log.push('Already running.')
            return
        state['previewing'] = False  # stop any active preview
        log.push('Generating point cloud...')
        status.set_text('Running...')
        status.style('color: #facc15')
        pc_progress.set_visibility(True)
        pc_progress_label.set_visibility(True)
        pc_progress.set_value(0)

        def on_progress(processed, total_samples):
            valueF = f'{processed / total_samples:.1%}'
            pc_progress.set_value(valueF)
            pc_progress_label.set_text(f'Frame {processed} / {total_samples}')

        def generate_point_cloud():
            try:
                pcg.generate_from_video(RGBD_VIDEO_DIR, on_progress=on_progress)
                log.push('Point cloud generation complete!')
            except Exception as e:
                log.push(f'Error: {e}')
            finally:
                state['running'] = False
                status.set_text('Ready')
                status.style('color: #69fdb3')

        state['running'] = True
        threading.Thread(target=generate_point_cloud, daemon=True).start()

    def preview_point_cloud():
        rgb_dir = os.path.join(RGBD_VIDEO_DIR, 'rgb')
        if not os.path.exists(rgb_dir):
            log.push('RGB source directory not found.')
            return
        if state['previewing']:
            log.push('Preview already playing.')
            return

        allowed_ext = ('.jpg', '.jpeg', '.png')
        image_files = sorted(
            [f for f in os.listdir(rgb_dir) if f.lower().endswith(allowed_ext)],
            key=lambda n: float(os.path.splitext(n)[0]) if os.path.splitext(n)[0].replace('.', '', 1).isdigit() else n
        )
        if not image_files:
            log.push('No frames found in rgb/ directory.')
            return

        log.push(f'Playing {len(image_files)} frames in app...')
        state['previewing'] = True

        def playback_loop():
            UI_MAX_FPS = 30
            interval = 1.0 / UI_MAX_FPS
            for img_file in image_files:
                if not state['previewing']:
                    break
                frame = cv.imread(os.path.join(rgb_dir, img_file))
                if frame is None:
                    continue
                try:
                    video_image.set_source(frame_to_data_url(frame))
                except Exception:
                    break
                time.sleep(interval)
            state['previewing'] = False
            try:
                log.push('Preview finished.')
            except Exception:
                pass

        threading.Thread(target=playback_loop, daemon=True).start()

    def stop_preview():
        if state['previewing']:
            state['previewing'] = False
            log.push('Preview stopped.')
        else:
            log.push('Nothing is playing.')

    with ui.element('div').classes('app-container'):

        build_sidebar('reconstruction')

        with ui.element('div').classes('main-panel'):

            with ui.element('div').classes('top-part'):
                with ui.element('div').classes('video-box'):
                    ui.label('3D Reconstruction').classes('video-label')
                    video_image = ui.interactive_image('').style(
                        'width:100%; height:100%; object-fit:contain;'
                    )

                with ui.element('div').classes('control-panel'):
                    with ui.element('div').classes('run-stop'):
                        with ui.element('button').classes('btn-run').on('click', run_point_cloud):
                            ui.label('Generate')
                            ui.image('./ui_IMG/cube.svg').style('width: 15px;')
                        with ui.element('button').classes('btn-stop').on('click', preview_point_cloud):
                            ui.label('Preview Video')
                            ui.image('./ui_IMG/play.svg').style('width: 15px;')
                        with ui.element('button').classes('btn-stop').on('click', stop_preview):
                            ui.label('Stop Preview')
                            ui.image('./ui_IMG/stop.svg').style('width: 15px;')

            with ui.element('div').classes('bottom-section'):
                with ui.element('div').classes('card status-card'):
                    ui.label('Status').classes('card-header')
                    status = ui.label('Ready').style(
                        'color: #69fdb3; font-weight: bold; font-size: 1rem;'
                    )

                with ui.element('div').classes('card progress-card'):
                    ui.label('Frame Progress').classes('card-header')
                    pc_progress_label = ui.label('').style('color: #242527; font-size: 0.875rem;')
                    pc_progress_label.set_visibility(False)
                    pc_progress = ui.linear_progress(0).style('width: 100%;')
                    pc_progress.set_visibility(False)

                with ui.element('div').classes('card log-card'):
                    log = ui.log(max_lines=30).style(
                        'height: 120px; color: black;'
                        ' font-family: monospace; font-size: 0.875rem; background: transparent; border: none;'
                    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
app.native.window_args['resizable'] = False
app.native.settings['ALLOW_DOWNLOADS'] = True


ui.run(native=True, window_size=(1175, 575), title='COMP 4990 — Computer Vision')
