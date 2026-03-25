import os
import cv2 as cv
import time
import threading
import base64
import json

from src.detector import Detector
from src.video_handler import VideoHandler
from src.utils import draw_detections
from src.tum_generator import TUMGenerator as tum
from src.live_generator import LiveGenerator
import webview

from nicegui import app, ui

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SETTINGS_PATH = os.path.join(PROJECT_ROOT, 'settings.json')

SETTINGS_DEFAULT = {
    'local_video_path': os.path.join(PROJECT_ROOT, 'data', 'input_videos', 'videoTest.mp4'),
    'rtsp_url': "rtsp://192.168.2.23:8554/stream",
    'output_path': os.path.join(PROJECT_ROOT, 'data', 'output_videos', 'output.mp4'),
    'rgbd_video_dir': os.path.join(PROJECT_ROOT, 'data', 'input_RGBD_videos', 'rgbd_dataset_freiburg1_room'),
    's_tolerence': 0.5,
}
SETTINGS_PATH = os.path.join(PROJECT_ROOT, 'settings.json')

SETTINGS = SETTINGS_DEFAULT.copy()

OUTPUT_3D_MODELS_PATH = os.path.join(PROJECT_ROOT, 'data', 'output_3D_models')


_shutting_down = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def frame_to_data_url(frame: cv.typing.MatLike) -> str:
    """encode an OpenCV frame as a JPEG data URL for UI display."""
    _, buf = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode('utf-8')
    return f'data:image/jpeg;base64,{b64}'


def load_settings() -> None:
    """Load settings from JSON file, or create it with defaults if not found."""
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, 'r') as f:
                loaded = json.load(f)
                SETTINGS.update(loaded)
                print(f'Settings loaded from {SETTINGS_PATH}')
                return
        except Exception as e:
            print(f'Error loading settings: {e}')
    else:
        save_settings()
        print(f'Settings file created with default settings at {SETTINGS_PATH}')


def save_settings() -> None:
    """Save current settings to JSON file."""
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(SETTINGS, f, indent=4)

load_settings()

SOURCE_MAP = {
    'Local Video File': SETTINGS['local_video_path'],
    'Laptop Webcam': 0,
    'Phone Camera (RTSP)': SETTINGS['rtsp_url'],
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
CSS = '''<style>
  * { box-sizing: border-box; }
  body, .q-page, .nicegui-content { padding: 0 !important; margin: 0 !important; }
  .q-page { background-color: #242527 !important; }

  /* Scrollbar */
  * { scrollbar-width: thin; scrollbar-color: #242527 transparent; }
  *::-webkit-scrollbar { width: 6px; height: 6px; }
  *::-webkit-scrollbar-track { background: transparent; }
  *::-webkit-scrollbar-thumb { background-color: #242527; border-radius: 3px; border: none; }
  *::-webkit-scrollbar-corner { background: transparent; }
  *::-webkit-scrollbar-button { display: none; height: 0; width: 0; }

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
    flex: 1;
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
    width: 1140px;
  }
  .video-box {
    width: 800px;
    height: 450px;
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
    height: 450px;
    width: 320px;
  }
  .run-stop { display: flex; flex-direction: column; gap: 10px; }
  .btn-run, .btn-stop {
    min-width: 320px;
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
  .bottom-section { display: flex; gap: 10px; margin: 10px; max-width: 1130px; }
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

  /* Settings page */
  .settings-main {
    background-color: #dedede;
    border-radius: 20px;
    margin: 5px 0;
    flex: 1;
    overflow-x: hidden;
    display: flex;
    flex-direction: column;
    max-width: 930px;
    width: 930px;
  }
  .settings-title {
    font-size: 2rem;
    font-weight: 800;
    color: black;
    margin: 24px 0 0 24px;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .settings-subtitle {
    font-size: 0.8rem;
    color: rgba(0,0,0,0.5);
    margin-bottom: 24px;
    margin-left: 24px;
  }
  .settings-card {
    background-color: white;
    width: calc(100% - 48px);
    border-radius: 12px;
    padding: 24px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px 32px;
    margin-bottom: 16px;
    margin: 0 24px 16px 24px;
  }
  .settings-field { display: flex; flex-direction: column; gap: 6px; }
  .settings-field-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(0,0,0,0.5);
  }
  .settings-field-row { display: flex; gap: 8px; align-items: center; }
  .settings-input-wrap { flex: 1; }
  .settings-input-wrap .q-field__control {
    background: #2a2d33 !important;
    border-radius: 6px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
  }
  .settings-input-wrap .q-field__native { color: white !important; font-size: 13px; padding-left: 8px !important; }
  .settings-input-wrap .q-field__bottom { display: none; }
  .settings-input-wrap .q-field__before, .settings-input-wrap .q-field__after { display: none; }

  .rtsp-input-wrap { flex: 1; }
  .rtsp-input-wrap .q-field__control {
    background: #f5f5f3 !important;
    border-radius: 6px !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
  }
  .rtsp-input-wrap .q-field__native { color: #111 !important; font-size: 13px; padding-left: 8px !important; }
  .rtsp-input-wrap .q-field__bottom { display: none; }
  .rtsp-input-wrap .q-field__before, .rtsp-input-wrap .q-field__after { display: none; }
  .rtsp-input-wrap.q-field--focused .q-field__control,
  .rtsp-input-wrap .q-field--focused .q-field__control { border-color: #bcb1f3 !important; }
  .rtsp-input-wrap .q-field__control:after { background: #bcb1f3 !important; }
  .rtsp-input-wrap.q-field--highlighted .q-field__control:after,
  .rtsp-input-wrap .q-field--highlighted .q-field__control:after { background: #bcb1f3 !important; }
  .btn-browse {
    background-color: #bcb1f3;
    color: #111;
    border: none;
    border-radius: 6px;
    padding: 0 16px;
    align-self: stretch;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    white-space: nowrap;
    transition: opacity 0.2s;
  }
  .btn-browse:hover { opacity: 0.75; }
  .btn-browse label { pointer-events: none; color: #111 !important; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
  .confidence-card {
    background-color: white;
    border-radius: 12px;
    padding: 24px 28px;
    margin: 0 24px 16px 24px;
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .confidence-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }
  .confidence-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #111;
  }
  .confidence-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #111;
  }
  .confidence-subtitle {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(0,0,0,0.4);
  }
  .confidence-ticks {
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: rgba(0,0,0,0.35);
    margin-top: 4px;
  }
  .settings-actions {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    margin-top: auto;
    padding-top: 8px;
    padding-bottom: 24px;
  }
  .btn-return {
    background-color: #bcb1f3;
    border: none;
    color: #111;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    padding: 0 28px;
    height: 44px;
    border-radius: 8px;
    transition: opacity 0.2s;
  }
  .btn-return:hover { opacity: 0.75; }
  .btn-return label { pointer-events: none; color: #111 !important; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; }
  .btn-save {
    background-color: #69fdb3;
    border: none;
    color: #111;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    padding: 0 28px;
    height: 44px;
    border-radius: 8px;
    transition: opacity 0.2s;
    margin-right: 24px;
  }
  .btn-save:hover { opacity: 0.75; }
  .btn-save label { pointer-events: none; color: #111 !important; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; }
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
            with ui.element('div').classes('nav-pill').on(
                'click', lambda: ui.navigate.to('/settings')
            ).style('margin-top: auto;'):
                ui.icon('settings').style('font-size:16px;')
                ui.label('Settings')

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@ui.page('/')
def main_page():
    state = {'running': False, 'source': 'Local Video File', 'tolerance': SETTINGS['s_tolerence']}

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
                handler = VideoHandler(source, SETTINGS['output_path'])
                safe_log(f'Connected: {handler.width}x{handler.height} @ {handler.fps} FPS')

                DETECT_EVERY_N = 3
                UI_MAX_FPS = 20
                ui_interval = 1.0 / UI_MAX_FPS
                frame_count = 0
                last_results = None
                last_ui_update = 0.0

                frame_interval = 1.0 / handler.fps if not handler.is_live else 0.0

                while state['running'] and not _shutting_down:
                    frame_start = time.monotonic()
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

                    if not handler.is_live:
                        elapsed = time.monotonic() - frame_start
                        sleep_time = frame_interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                handler.release()
                safe_log(f'Saved to: {SETTINGS["output_path"]}')
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

            with ui.element('div').classes('bottom-section'):
                with ui.element('div').classes('card status-card'):
                    ui.label('Status').classes('confidence-subtitle')
                    status = ui.label('Ready').style(
                        'color: #69fdb3 ; font-weight: bold; font-size: 1rem;'
                    )

                with ui.element('div').classes('card progress-card'):
                    ui.label('TRACKING TOLERENCE').classes('confidence-subtitle')
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
    state = {'running': False, 'previewing': False, 'source': 'TUM Dataset', 'live_scanning': False}
    live_gen = LiveGenerator(output_path=OUTPUT_3D_MODELS_PATH)

    ui.add_head_html(CSS)

    # ── Source & UI Handlers ──────────────────────────────────────────────────
    def update_source_ui(selected: str) -> None:
        for name, btn in source_btns.items():
            if name == selected:
                btn.classes(add='active-src')
            else:
                btn.classes(remove='active-src')

        is_tum = selected == 'TUM Dataset'
        btn_run_tum.set_visibility(is_tum)
        btn_toggle_preview.set_visibility(is_tum)
        btn_start_live.set_visibility(not is_tum)

    def select_source(name: str) -> None:
        if state['running'] or state['live_scanning'] or state['previewing']:
            log.push("Stop active processes before changing source.")
            return
        state['source'] = name
        update_source_ui(name)

    # ── Live Generation Polling Loop ──────────────────────────────────────────
    def update_reconstruction_ui():
        """Polling function to securely update the UI from the main thread."""

        # 1. Update the live scanning progress
        if state['live_scanning']:
            pc_progress.set_visibility(True)
            pc_progress_label.set_visibility(True)
            pos = live_gen.current_pos
            pc_progress_label.set_text(
                f"Frames: {live_gen.count} | Pos: [{pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}]"
            )
            pc_progress.set_value((live_gen.count % 100) / 100)

            # Show live preview frame if available
            if hasattr(live_gen, 'latest_frame') and live_gen.latest_frame is not None:
                try:
                    video_image.set_source(frame_to_data_url(live_gen.latest_frame))
                except Exception:
                    pass

        # 2. Check if the background extraction has finished
        if hasattr(live_gen, 'finished_msg') and live_gen.finished_msg is not None:
            success, msg = live_gen.finished_msg
            log.push(msg)

            # Reset UI state
            status.set_text('Ready')
            status.style('color: #69fdb3')
            pc_progress.set_visibility(False)
            state['live_scanning'] = False

            # Clear the message to avoid duplicate processing
            live_gen.finished_msg = None

            # Register the main-thread timer (runs 10 times a second)

    ui.timer(0.1, update_reconstruction_ui)

    def start_live_scan():
        if state['live_scanning']: return
        success, msg = live_gen.start_scan()
        if success:
            state['live_scanning'] = True
            log.push('Live scan started. Stop from the iPhone app when done.')
            status.set_text('Scanning...')
            status.style('color: #facc15')
            pc_progress.set_visibility(True)
            pc_progress_label.set_visibility(True)
        else:
            log.push(f'Failed to connect: {msg}')

    # ── TUM Dataset Actions ───────────────────────────────────────────────────
    def run_point_cloud():
        if state['running']:
            log.push('Already running.')
            return
        state['previewing'] = False
        reset_preview_button()
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
                tum.generate_from_tum(SETTINGS['rgbd_video_dir'], output_path=OUTPUT_3D_MODELS_PATH,
                                      on_progress=on_progress)
                log.push('Point cloud generation complete!')
            except Exception as e:
                log.push(f'Error: {e}')
            finally:
                state['running'] = False

                def ui_reset():
                    status.set_text('Ready')
                    status.style('color: #69fdb3')

                ui.timer(0.1, ui_reset, once=True)

        state['running'] = True
        threading.Thread(target=generate_point_cloud, daemon=True).start()

    def reset_preview_button():
        preview_label.set_text('Preview Video')
        preview_icon.set_source('./ui_IMG/play.svg')

    def toggle_preview():
        if state['previewing']:
            state['previewing'] = False
            log.push('Preview stopped.')
            reset_preview_button()
        else:
            rgb_dir = os.path.join(SETTINGS['rgbd_video_dir'], 'rgb')
            if not os.path.exists(rgb_dir):
                log.push('RGB source directory not found.')
                return

            allowed_ext = ('.jpg', '.jpeg', '.png')
            image_files = sorted(
                [f for f in os.listdir(rgb_dir) if f.lower().endswith(allowed_ext)],
                key=lambda n: float(os.path.splitext(n)[0]) if os.path.splitext(n)[0].replace('.', '',
                                                                                              1).isdigit() else n
            )
            if not image_files:
                log.push('No frames found in rgb/ directory.')
                return

            log.push(f'Playing {len(image_files)} frames in app...')
            state['previewing'] = True
            preview_label.set_text('Stop Preview')
            preview_icon.set_source('./ui_IMG/stop.svg')

            def playback_loop():
                UI_MAX_FPS = 30
                interval = 1.0 / UI_MAX_FPS
                for img_file in image_files:
                    if not state['previewing'] or _shutting_down:
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

                def finish_ui():
                    reset_preview_button()
                    try:
                        log.push('Preview finished.')
                    except Exception:
                        pass

                ui.timer(0.1, finish_ui, once=True)

            threading.Thread(target=playback_loop, daemon=True).start()

    # ── Layout ────────────────────────────────────────────────────────────────
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
                        btn_run_tum = ui.element('button').classes('btn-run').on('click', run_point_cloud)
                        with btn_run_tum:
                            ui.label('Generate')
                            ui.image('./ui_IMG/cube.svg').style('width: 15px;')

                        btn_toggle_preview = ui.element('button').classes('btn-stop').on('click', toggle_preview)
                        with btn_toggle_preview:
                            preview_label = ui.label('Preview Video')
                            preview_icon = ui.image('./ui_IMG/play.svg').style('width: 15px;')

                        btn_start_live = ui.element('button').classes('btn-run').on('click', start_live_scan)
                        with btn_start_live:
                            ui.label('Start Live Scan')
                            ui.image('./ui_IMG/cube.svg').style('width: 15px;')

                    with ui.element('div').classes('source-section'):
                        ui.label('Data Source:').classes('source-title')
                        source_btns: dict = {}
                        for src_name in ['TUM Dataset', 'Live Stream']:
                            btn_el = ui.element('button').classes('source-btn').on(
                                'click', lambda n=src_name: select_source(n)
                            )
                            with btn_el:
                                ui.label(src_name)
                            source_btns[src_name] = btn_el

                        update_source_ui(state['source'])

            with ui.element('div').classes('bottom-section'):
                with ui.element('div').classes('card status-card'):
                    ui.label('Status').classes('confidence-subtitle')
                    status = ui.label('Ready').style(
                        'color: #69fdb3; font-weight: bold; font-size: 1rem;'
                    )

                with ui.element('div').classes('card progress-card'):
                    ui.label('Frame Progress').classes('confidence-subtitle')
                    pc_progress_label = ui.label('').style('color: #242527; font-size: 0.875rem;')
                    pc_progress_label.set_visibility(False)
                    pc_progress = ui.linear_progress(0).style('width: 100%;')
                    pc_progress.set_visibility(False)

                with ui.element('div').classes('card log-card'):
                    log = ui.log(max_lines=30).style(
                        'height: 120px; color: black;'
                        ' font-family: monospace; font-size: 0.875rem; background: transparent; border: none;'
                    )

@ui.page('/settings')
def settings_page():
    def save_and_back():
        SETTINGS['local_video_path'] = state['vidPath']
        SOURCE_MAP['Local Video File'] = state['vidPath']
        SETTINGS['rtsp_url'] = state['rtspURL']
        SOURCE_MAP['Phone Camera (RTSP)'] = state['rtspURL']
        SETTINGS['output_path'] = state['outputPath']
        SETTINGS['rgbd_video_dir'] = state['rgbdVideoDir']
        SETTINGS['s_tolerence'] = state['tolerance']

        save_settings()
        ui.navigate.to('/')

    state = {
        'vidPath': SETTINGS['local_video_path'],
        'rtspURL': SETTINGS['rtsp_url'],
        'outputPath': SETTINGS['output_path'],
        'rgbdVideoDir': SETTINGS['rgbd_video_dir'],
        'tolerance': SETTINGS['s_tolerence'],
    }
    async def pick_video():
        result = await app.native.main_window.create_file_dialog(
            allow_multiple=False,
            file_types=('Video Files (*.mp4;*.avi;*.mov;*.mkv)',)
        )
        if result:
            state['vidPath'] = result[0]

    async def pick_dir(type: str):
        result = await app.native.main_window.create_file_dialog(
            20,  # webview.FOLDER_DIALOG
            allow_multiple=False,
        )
        if result and type == 'rgbd':
            state['rgbdVideoDir'] = result[0]
        elif result and type == 'output':
            state['outputPath'] = os.path.join(result[0], 'data', 'output_videos', 'output.mp4')

    ui.add_head_html(CSS)

    with ui.element('div').classes('app-container'):
        build_sidebar('')

        with ui.element('div').classes('settings-main'):
            ui.label('Application Parameters').classes('settings-title')
            ui.label('Manage your video sources, output paths, and detection settings.').classes('settings-subtitle')

            with ui.element('div').classes('settings-card'):

                with ui.element('div').classes('settings-field'):
                    ui.label('Local Video Path').classes('settings-field-label')
                    with ui.element('div').classes('settings-field-row'):
                        ui.input().classes('rtsp-input-wrap').bind_value(state, 'vidPath')
                        with ui.element('button').classes('btn-browse').on('click', pick_video):
                            ui.label('Browse')

                with ui.element('div').classes('settings-field'):
                    ui.label('RTSP URL').classes('settings-field-label')
                    ui.input().classes('rtsp-input-wrap').bind_value(state, 'rtspURL')

                with ui.element('div').classes('settings-field'):
                    ui.label('Output Video Path').classes('settings-field-label')
                    with ui.element('div').classes('settings-field-row'):
                        ui.input().classes('rtsp-input-wrap').bind_value(state, 'outputPath')
                        with ui.element('button').classes('btn-browse').on('click', lambda: pick_dir('output')):
                            ui.label('Browse')

                with ui.element('div').classes('settings-field'):
                    ui.label('RGBD Directory').classes('settings-field-label')
                    with ui.element('div').classes('settings-field-row'):
                        ui.input().classes('rtsp-input-wrap').bind_value(state, 'rgbdVideoDir')
                        with ui.element('button').classes('btn-browse').on('click', lambda: pick_dir('rgbd')):
                            ui.label('Browse')

            with ui.element('div').classes('confidence-card'):
                with ui.element('div').classes('confidence-header'):
                    ui.label('Confidence Threshold').classes('confidence-title')
                    confidence_display = ui.label(f"{state['tolerance']:.2f}").classes('confidence-value')
                ui.label('Minimum probability for object detection events').classes('confidence-subtitle').style('margin-bottom: 10px;')
                ui.slider(min=0.0, max=1.0, step=0.01).classes('tolerance-slider').bind_value(state, 'tolerance').on(
                    'update:model-value',
                    lambda e: confidence_display.set_text(f'{e.args:.2f}')
                )
                with ui.element('div').classes('confidence-ticks'):
                    ui.label('0.0 (Loose)')
                    ui.label('0.5 (Neutral)')
                    ui.label('1.0 (Strict)')

            with ui.element('div').classes('settings-actions'):
                with ui.element('button').classes('btn-return').on('click', lambda: ui.navigate.to('/')):
                    ui.label('Return').style('color: black;')
                with ui.element('button').classes('btn-save').on('click', save_and_back):
                    ui.label('Save Configuration')


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ in {"__main__", "__mp_main__"}:

    @app.on_shutdown
    async def _on_shutdown():
        global _shutting_down
        _shutting_down = True

    app.native.window_args['resizable'] = False
    app.native.settings['ALLOW_DOWNLOADS'] = True
    ui.run(native=True, window_size=(1400, 800), title='COMP 4990 — Computer Vision')