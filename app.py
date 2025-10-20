import streamlit as st
import threading
import queue
import time
import datetime
import io
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import librosa
from realtime_solo import RealTimeSolo
from record_sound import ChunkRecorder

# --- Page Config ---
st.set_page_config(
    page_title="SilentSignals",
    page_icon="🚨",
    layout="wide",
)

# --- Constants for Session State ---
STATE_RUNNING = 'pipeline_running'
STATE_THREAD = 'pipeline_thread'
STATE_STOP_EVENT = 'stop_event'
STATE_ALERTS = 'alerts'
STATE_ALERT_QUEUE = 'alert_queue'
STATE_AUDIO_TO_PROCESS = 'audio_data_to_process'
STATE_AUDIO_LEVEL = 'current_audio_level'
STATE_FILE_PROGRESS = 'file_progress'
STATE_ALERT_HISTORY = 'alert_history'

# --- Dataclasses for Queue Communication ---
@dataclass
class ErrorMessage:
    text: str

@dataclass
class LevelMessage:
    level: float

@dataclass
class ProgressMessage:
    progress: float

# Alert messages are dicts returned from solo.process_chunk()
QueueItem = ErrorMessage | LevelMessage | ProgressMessage | Dict[str, Any]


# --- Best Practice: Centralized State Management ---
def init_session_state() -> None:
    """Initializes all required keys in Streamlit's session state."""
    state_defaults = {
        STATE_RUNNING: False,
        STATE_THREAD: None,
        STATE_STOP_EVENT: None,
        STATE_ALERTS: {},
        STATE_ALERT_QUEUE: queue.Queue(),
        STATE_AUDIO_TO_PROCESS: None,
        STATE_AUDIO_LEVEL: 0.0,
        STATE_FILE_PROGRESS: 0.0,
        STATE_ALERT_HISTORY: [],
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def load_model() -> Optional[RealTimeSolo]:
    """
    Loads the RealTimeSolo model once and caches it.
    Returns None if loading fails.
    """
    print("Attempting to load AI model...")
    try:
        model = RealTimeSolo("./configs/config.yaml")
        print("AI model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load AI model: {e}")
        return None

# --- Background Pipeline Tasks ---

def mic_pipeline(
    stop_event: threading.Event,
    alert_queue: queue.Queue[QueueItem],
    model: RealTimeSolo
) -> None:
    """Handles microphone recording and real-time processing."""
    try:
        model.start_session()

        def handle_chunk(chunk: np.ndarray) -> None:
            # Calculate audio level (RMS) and put on queue
            rms = np.sqrt(np.mean(chunk**2) + 1e-10)
            level = float(np.clip(rms * 10, 0, 1))
            alert_queue.put(LevelMessage(level=level))

            # Process for alerts
            result = model.process_chunk(chunk, chunk_sr=model.sr)
            if result:
                alert_queue.put(result)

        rec = ChunkRecorder(
            sr=model.sr,
            channels=1,
            chunk_size_sec=model.chunk_ms / 1000.0,
            on_chunk=handle_chunk
        )
        rec.start()
        stop_event.wait()  # Wait until the stop event is set
        rec.stop()
        model.stop_session()

    except Exception as e:
        alert_queue.put(ErrorMessage(text=f"Microphone Pipeline Error: {e}"))


def file_pipeline(
    stop_event: threading.Event,
    alert_queue: queue.Queue[QueueItem],
    audio_bytes: bytes,
    model: RealTimeSolo
) -> None:
    """Handles audio file processing in a background thread."""
    try:
        model.start_session()
        try:
            # Load audio data from in-memory bytes
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=model.sr, mono=True)
            total_samples = len(y)
        except Exception as e:
            alert_queue.put(ErrorMessage(text=f"Could not process audio file: {e}"))
            model.stop_session()
            return

        hop = int(model.sr * (model.chunk_ms / 1000.0))
        
        for i in range(0, total_samples, hop):
            if stop_event.is_set():
                break
                
            progress_percent = (i / total_samples)
            alert_queue.put(ProgressMessage(progress=progress_percent))
            
            chunk = y[i:i + hop]
            if len(chunk) < hop:
                # Pad the final chunk if it's too short
                chunk = np.pad(chunk, (0, hop - len(chunk)))
            
            result = model.process_chunk(chunk, chunk_sr=model.sr)
            if result:
                alert_queue.put(result)
                
        alert_queue.put(ProgressMessage(progress=1.0)) # Signal completion
        model.stop_session()
        
    except Exception as e:
        alert_queue.put(ErrorMessage(text=f"File Processing Pipeline Error: {e}"))


# --- UI Drawing Functions ---

def draw_control_panel(model: RealTimeSolo) -> None:
    """Draws the main control panel (Mic/File mode) and start/stop buttons."""
    
    mode = st.radio("Audio Source", ["Microphone", "Audio File"], index=0, horizontal=True)

    if mode == "Microphone":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🎤 Start Recording", disabled=st.session_state[STATE_RUNNING]):
                st.session_state[STATE_ALERTS].clear()
                st.session_state[STATE_ALERT_HISTORY].clear()
                
                stop_event = threading.Event()
                thread = threading.Thread(
                    target=mic_pipeline,
                    args=(stop_event, st.session_state[STATE_ALERT_QUEUE], model),
                    daemon=True
                )
                thread.start()

                st.session_state[STATE_STOP_EVENT] = stop_event
                st.session_state[STATE_THREAD] = thread
                st.session_state[STATE_RUNNING] = True
                st.rerun()

        with col2:
            if st.button("❌ Stop Recording", disabled=not st.session_state[STATE_RUNNING]):
                if st.session_state[STATE_STOP_EVENT]:
                    st.session_state[STATE_STOP_EVENT].set()
                st.session_state[STATE_RUNNING] = False
                st.session_state[STATE_AUDIO_LEVEL] = 0.0
                st.rerun()

        with col3:
            if st.button("🗑️ Clear All Alerts"):
                st.session_state[STATE_ALERTS].clear()
                st.session_state[STATE_ALERT_HISTORY].clear()
                st.rerun()

    else:  # Audio File mode
        file_col, run_col, reset_col = st.columns([2, 1, 1])
        with file_col:
            uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
            
            if uploaded_file:
                st.session_state[STATE_AUDIO_TO_PROCESS] = uploaded_file.read()
            elif STATE_AUDIO_TO_PROCESS in st.session_state:
                 st.session_state[STATE_AUDIO_TO_PROCESS] = None

        with run_col:
            can_process = (st.session_state[STATE_AUDIO_TO_PROCESS] is not None)
            if st.button("▶️ Process File", disabled=(not can_process or st.session_state[STATE_RUNNING])):
                st.session_state[STATE_ALERTS].clear()
                st.session_state[STATE_ALERT_HISTORY].clear()
                st.session_state[STATE_FILE_PROGRESS] = 0.0
                
                stop_event = threading.Event()
                audio_data = st.session_state[STATE_AUDIO_TO_PROCESS]
                thread = threading.Thread(
                    target=file_pipeline,
                    args=(stop_event, st.session_state[STATE_ALERT_QUEUE], audio_data, model),
                    daemon=True
                )
                thread.start()

                st.session_state[STATE_STOP_EVENT] = stop_event
                st.session_state[STATE_THREAD] = thread
                st.session_state[STATE_RUNNING] = True
                st.rerun()

        with reset_col:
            if st.button("🗑️ Clear All Alerts"):
                st.session_state[STATE_ALERTS].clear()
                st.session_state[STATE_ALERT_HISTORY].clear()
                st.rerun()

def process_alert_queue() -> None:
    """Processes all items currently in the alert queue without blocking."""
    while not st.session_state[STATE_ALERT_QUEUE].empty():
        item = st.session_state[STATE_ALERT_QUEUE].get_nowait()
        
        if isinstance(item, ErrorMessage):
            st.error(item.text)
            st.session_state[STATE_RUNNING] = False
        
        elif isinstance(item, LevelMessage):
            st.session_state[STATE_AUDIO_LEVEL] = item.level
            
        elif isinstance(item, ProgressMessage):
            st.session_state[STATE_FILE_PROGRESS] = item.progress
        
        # This handles the alert dict
        elif isinstance(item, dict) and "message" in item:
            alert = item
            alert_key = alert.get("YAMNet", {}).get("top_label", "Unknown Event")

            if alert_key in st.session_state[STATE_ALERTS]:
                # Update existing alert
                st.session_state[STATE_ALERTS][alert_key]['count'] += 1
                st.session_state[STATE_ALERTS][alert_key]['alert_obj'] = alert
            else:
                # Create new alert
                tier = alert.get("tier", 1)
                if tier >= 2:
                    icon = "🚨" if tier == 3 else "⚠️"
                    st.toast(f"New Alert: {alert.get('message', '...')}", icon=icon)
                
                    if tier == 3:
                        try:
                            # Play sound for critical alerts
                            st.audio("alert.wav", autoplay=True) 
                        except Exception as e:
                            print(f"Could not play alert.wav: {e}")

                st.session_state[STATE_ALERTS][alert_key] = {
                    'alert_obj': alert,
                    'count': 1,
                    'first_seen': time.time(),
                    'id': str(uuid.uuid4())
                }

def draw_status_bar() -> None:
    """Draws the main status bar (Running/Stopped) and progress bars."""
    with st.container(border=True):
        if st.session_state[STATE_RUNNING]:
            thread: Optional[threading.Thread] = st.session_state[STATE_THREAD]
            if thread and thread.is_alive():
                st.markdown("**(Status: 🟢 Running)**")
            else:
                st.session_state[STATE_RUNNING] = False
                st.markdown("**(Status: 🏁 Finished)**")
        else:
            st.markdown("**(Status: 🔴 Stopped)**")

        # Reset indicators if not running
        if not st.session_state[STATE_RUNNING]:
            st.session_state[STATE_FILE_PROGRESS] = 0.0
            st.session_state[STATE_AUDIO_LEVEL] = 0.0

        # Show Live Audio Level for Microphone mode
        if st.session_state[STATE_AUDIO_LEVEL] > 0:
            st.progress(st.session_state[STATE_AUDIO_LEVEL], text="Live Audio Level")

        # Show Progress Bar for File mode (only when running)
        if st.session_state[STATE_FILE_PROGRESS] > 0:
            st.progress(st.session_state[STATE_FILE_PROGRESS], text="File Processing Progress")

def draw_active_alerts() -> None:
    """Draws the list of currently active, un-acknowledged alerts."""
    if not st.session_state[STATE_ALERTS]:
        st.write("✅ No active alerts at the moment.")
        return

    sorted_alerts = sorted(
        st.session_state[STATE_ALERTS].items(),
        key=lambda item: item[1]['first_seen'],
        reverse=True
    )
    
    for alert_key, alert_data in sorted_alerts:
        with st.container(border=True):
            alert = alert_data['alert_obj']
            alert_id = alert_data['id']
            count = alert_data['count']
            tier = alert.get("tier", 1)
            message = alert.get("message", "No message")

            alert_text = f"**{message}**"
            if count > 1:
                alert_text += f" (Detected {count} times)"

            col1, col2 = st.columns([4, 1])
            
            with col1:
                if tier >= 3:
                    st.error(alert_text, icon="🚨")
                elif tier == 2:
                    st.warning(alert_text, icon="⚠️")
                else:
                    st.info(alert_text, icon="ℹ️")
                        
                with st.expander("See details"):
                    yamnet_label = alert.get("YAMNet", {}).get("top_label", "N/A")
                    yamnet_score = alert.get("YAMNet", {}).get("top_score", 0)
                    pann_label = alert.get("PANN", {}).get("top_label", "N/A")
                    pann_score = alert.get("PANN", {}).get("top_score", 0)
                    
                    st.markdown(f"""
                    - **YAMNet Prediction:** `{yamnet_label}` (Score: `{yamnet_score*100:.1f}%`)
                    - **PANN Prediction:** `{pann_label}` (Score: `{pann_score*100:.1f}%`)
                    """)
            
            with col2:
                # Acknowledge button moves alert to history
                if st.button("Acknowledge", key=alert_id):
                    timestamp = datetime.datetime.now().strftime("%I:%M:%S %p")
                    st.session_state[STATE_ALERT_HISTORY].insert(0, {
                        "text": alert_text,
                        "timestamp": timestamp,
                        "tier": tier
                    })
                    del st.session_state[STATE_ALERTS][alert_key]
                    st.rerun()

def draw_alert_history() -> None:
    """Draws the expander holding all acknowledged alerts."""
    if st.session_state[STATE_ALERT_HISTORY]:
        st.markdown("---")
        with st.expander("Acknowledged Alert History"):
            for item in st.session_state[STATE_ALERT_HISTORY]:
                st.markdown(f"*{item['timestamp']}* - {item['text']}")

def run_live_update() -> None:
    """
    If the pipeline is running, triggers a short sleep and an st.rerun()
    to create the "live" polling effect for the UI.
    """
    if st.session_state[STATE_RUNNING]:
        time.sleep(0.1)  # Poll every 100ms
        st.rerun()

# --- Main App Execution ---

st.title("SilentSignals: Real-Time Event Based Alerts")

init_session_state()
solo_model = load_model()

if solo_model:
    # All main UI and logic functions are called here
    draw_control_panel(solo_model)
    process_alert_queue()
    draw_status_bar()

    st.markdown("---")
    st.subheader("Active Alerts:")
    draw_active_alerts()
    draw_alert_history()

    # This function must be called last
    run_live_update()
else:
    st.error("Model failed to load. The application cannot start.")
    st.warning("Please check your `config.yaml` file and model paths.")