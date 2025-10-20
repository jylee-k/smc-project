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
# Using constants prevents typos when accessing session state
STATE_RUNNING = 'pipeline_running'
STATE_THREAD = 'pipeline_thread'
STATE_STOP_EVENT = 'stop_event'
STATE_ACTIVE_ALERTS = 'active_alerts'
STATE_ALERT_QUEUE = 'alert_queue'
STATE_AUDIO_TO_PROCESS = 'audio_data_to_process'
STATE_AUDIO_LEVEL = 'current_audio_level'
STATE_FILE_PROGRESS = 'file_progress'
STATE_ALERT_HISTORY = 'alert_history'

# --- Dataclasses for Queue Communication ---
# Using dataclasses is safer and clearer than raw dicts
@dataclass
class ErrorMessage:
    text: str

@dataclass
class LevelMessage:
    level: float

@dataclass
class ProgressMessage:
    progress: float

# The queue can hold any of our defined message types or a dict from the model
QueueItem = ErrorMessage | LevelMessage | ProgressMessage | Dict[str, Any]


# --- Best Practice: Centralized State Management ---

def init_session_state() -> None:
    """Initializes all required keys in Streamlit's session state."""
    state_defaults = {
        STATE_RUNNING: False,
        STATE_THREAD: None,
        STATE_STOP_EVENT: None,
        STATE_ACTIVE_ALERTS: {},
        STATE_ALERT_QUEUE: queue.Queue(),
        STATE_AUDIO_TO_PROCESS: None,
        STATE_AUDIO_LEVEL: 0.0,
        STATE_FILE_PROGRESS: 0.0,
        STATE_ALERT_HISTORY: [],
    }
    # Ensure all keys exist on first run
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
    """
    Handles microphone recording and real-time processing in a separate thread.
    """
    try:
        model.start_session()

        def handle_chunk(chunk: np.ndarray) -> None:
            """Callback function for the ChunkRecorder."""
            # Calculate and queue the audio level for the UI
            rms = np.sqrt(np.mean(chunk**2) + 1e-10)
            level = float(np.clip(rms * 10, 0, 1))
            alert_queue.put(LevelMessage(level=level))

            # Process the chunk for events and queue the result
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
        stop_event.wait()  # Wait for the main thread to signal a stop
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
    """Handles audio file processing in a separate thread."""
    try:
        model.start_session()
        try:
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
            
            # Send progress updates to the UI
            progress_percent = (i / total_samples)
            alert_queue.put(ProgressMessage(progress=progress_percent))
            
            chunk = y[i:i + hop]
            if len(chunk) < hop:
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
                # Clear previous session data
                st.session_state[STATE_ACTIVE_ALERTS].clear()
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
                # Signal the thread to stop and update state
                if st.session_state[STATE_STOP_EVENT]:
                    st.session_state[STATE_STOP_EVENT].set()
                st.session_state[STATE_RUNNING] = False
                st.session_state[STATE_AUDIO_LEVEL] = 0.0
                st.rerun() # Trigger the main loop's cleanup logic

        with col3:
            if st.button("🗑️ Clear All Alerts"):
                # Clear all alerts without stopping the pipeline
                st.session_state[STATE_ACTIVE_ALERTS].clear()
                st.session_state[STATE_ALERT_HISTORY].clear()
                # No rerun needed; live_update will handle it

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
                st.session_state[STATE_ACTIVE_ALERTS].clear()
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
                st.session_state[STATE_ACTIVE_ALERTS].clear()
                st.session_state[STATE_ALERT_HISTORY].clear()
                # No rerun needed; live_update will handle it
                

def process_alert_queue() -> None:
    """Drains and processes all items from the thread-safe queue."""
    while not st.session_state[STATE_ALERT_QUEUE].empty():
        item = st.session_state[STATE_ALERT_QUEUE].get_nowait()
        
        if isinstance(item, ErrorMessage):
            st.error(item.text)
            st.session_state[STATE_RUNNING] = False
        
        elif isinstance(item, LevelMessage):
            st.session_state[STATE_AUDIO_LEVEL] = item.level
            
        elif isinstance(item, ProgressMessage):
            st.session_state[STATE_FILE_PROGRESS] = item.progress
        
        elif isinstance(item, dict) and "message" in item:
            # This is an alert from the model
            alert = item
            alert_key = alert.get("PANN", {}).get("top_label", "Unknown Event")
            tier = alert.get("tier", 1)

            if alert_key in st.session_state[STATE_ACTIVE_ALERTS]:
                # --- Logic for existing alerts ---
                # Update the count, scores, and last-seen time
                st.session_state[STATE_ACTIVE_ALERTS][alert_key]['count'] += 1
                st.session_state[STATE_ACTIVE_ALERTS][alert_key]['alert_obj'] = alert
                st.session_state[STATE_ACTIVE_ALERTS][alert_key]['last_seen'] = time.time()
            else:
                # --- Logic for new alerts ---
                # 1. Auto-archive any existing Tier 1 alerts
                keys_to_archive = []
                for key, data in st.session_state[STATE_ACTIVE_ALERTS].items():
                    if data['alert_obj'].get("tier", 1) == 1:
                        keys_to_archive.append(key)
                
                for key in keys_to_archive:
                    old_alert_data = st.session_state[STATE_ACTIVE_ALERTS].pop(key)
                    timestamp = datetime.datetime.fromtimestamp(old_alert_data['first_seen']).strftime("%I:%M:%S %p")
                    old_alert_text = f"**{old_alert_data['alert_obj'].get('message', '...')}** (Detected {old_alert_data['count']} times)"
                    
                    st.session_state[STATE_ALERT_HISTORY].insert(0, {
                        "text": old_alert_text,
                        "timestamp": timestamp,
                        "tier": 1
                    })

                # 2. Add the new alert (of any tier) to the active list
                st.session_state[STATE_ACTIVE_ALERTS][alert_key] = {
                    'alert_obj': alert,
                    'count': 1,
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'id': str(uuid.uuid4())
                }

                # 3. Show a toast notification for new high-priority alerts
                if tier >= 2:
                    icon = "🚨" if tier == 3 else "⚠️"
                    st.toast(f"New Alert: {alert.get('message', '...')}", icon=icon)

def draw_status_bar() -> None:
    """Draws the main status bar (Running/Stopped) and progress bars."""
    with st.container(border=True):
        if st.session_state[STATE_RUNNING]:
            thread: Optional[threading.Thread] = st.session_state[STATE_THREAD]
            if thread and thread.is_alive():
                st.markdown("**(Status: 🟢 Running)**")
            else:
                # The thread has finished, but state hasn't been cleaned up
                st.session_state[STATE_RUNNING] = False
                st.markdown("**(Status: 🏁 Finished)**")
        else:
            st.markdown("**(Status: 🔴 Stopped)**")

        # Clear progress bars if not running
        if not st.session_state[STATE_RUNNING]:
            st.session_state[STATE_FILE_PROGRESS] = 0.0
            st.session_state[STATE_AUDIO_LEVEL] = 0.0

        if st.session_state[STATE_AUDIO_LEVEL] > 0:
            st.progress(st.session_state[STATE_AUDIO_LEVEL], text="Live Audio Level")

        if st.session_state[STATE_FILE_PROGRESS] > 0:
            st.progress(st.session_state[STATE_FILE_PROGRESS], text="File Processing Progress")

def draw_active_alerts() -> None:
    """Draws all unacknowledged alerts (Tier 1, 2, and 3)."""
    
    if not st.session_state[STATE_ACTIVE_ALERTS]:
        st.write("✅ No active alerts at the moment.")
        return

    # Sort alerts by when they were first seen
    sorted_alerts = sorted(
        st.session_state[STATE_ACTIVE_ALERTS].items(),
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

            first_seen_str = datetime.datetime.fromtimestamp(alert_data['first_seen']).strftime("%I:%M:%S %p")
            last_seen_str = datetime.datetime.fromtimestamp(alert_data['last_seen']).strftime("%I:%M:%S %p")
            
            alert_text = f"**{message}**"
            if count > 1:
                alert_text += f" (Detected {count} times)"

            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Display color-coded alert based on tier
                if tier >= 3:
                    st.error(alert_text, icon="🚨")
                elif tier == 2:
                    st.warning(alert_text, icon="⚠️")
                else:
                    st.info(alert_text, icon="ℹ️")
                
                st.caption(f"First seen: {first_seen_str}  |  Last seen: {last_seen_str}")
                        
                with st.expander("See details", expanded=True):
                    yamnet_label = alert.get("YAMNet", {}).get("top_label", "N/A")
                    yamnet_score = alert.get("YAMNet", {}).get("top_score", 0)
                    pann_label = alert.get("PANN", {}).get("top_label", "N/A")
                    pann_score = alert.get("PANN", {}).get("top_score", 0)
                    
                    st.markdown(f"""
                    - **PANN Prediction:** `{pann_label}` (Score: `{pann_score*100:.1f}%`)
                    - **YAMNet Prediction:** `{yamnet_label}` (Score: `{yamnet_score*100:.1f}%`)
                    """)
            
            with col2:
                # Only show Acknowledge button for Tier 2+
                if tier >= 2:
                    if st.button("Acknowledge", key=alert_id):
                        # Move this alert to the history
                        timestamp = datetime.datetime.fromtimestamp(alert_data['first_seen']).strftime("%I:%M:%S %p")
                        st.session_state[STATE_ALERT_HISTORY].insert(0, {
                            "text": alert_text,
                            "timestamp": timestamp,
                            "tier": tier
                        })
                        # Remove it from the active list
                        del st.session_state[STATE_ACTIVE_ALERTS][alert_key]
                        # No rerun needed; live_update will handle it

def draw_tiered_alert_history() -> None:
    """Draws the acknowledged alert history, filtered into separate tiers."""
    
    if not st.session_state[STATE_ALERT_HISTORY]:
        st.markdown("---")
        st.write("✅ No acknowledged alerts in history.")
        return

    # Filter the single history list into three separate lists
    all_history = st.session_state[STATE_ALERT_HISTORY]
    tier3_history = [item for item in all_history if item.get("tier") == 3]
    tier2_history = [item for item in all_history if item.get("tier") == 2]
    tier1_history = [item for item in all_history if item.get("tier") == 1]

    st.markdown("---")
    
    # Tier 3 History Box
    with st.expander("Tier 3 Alert History", expanded=True):
        if not tier3_history:
            st.caption("No Tier 3 alerts in history.")
        else:
            for item in tier3_history:
                st.markdown(f"*{item['timestamp']}* - {item['text']}")

    # Tier 2 History Box
    with st.expander("Tier 2 Alert History", expanded=True):
        if not tier2_history:
            st.caption("No Tier 2 alerts in history.")
        else:
            for item in tier2_history:
                st.markdown(f"*{item['timestamp']}* - {item['text']}")
    
    # Tier 1 History Box
    with st.expander("Tier 1 Alert History"):
        if not tier1_history:
            st.caption("No Tier 1 alerts in history.")
        else:
            for item in tier1_history:
                st.markdown(f"*{item['timestamp']}* - {item['text']}")

def run_live_update() -> None:
    """
    If the pipeline is running, triggers a short sleep and an st.rerun()
    to create the "live" polling effect for the UI.
    """
    if st.session_state[STATE_RUNNING]:
        # Refresh rate is 300ms. This is slower than the 200ms model chunk size,
        # which prevents race conditions and stabilizes the UI.
        time.sleep(0.5)
        st.rerun()

# --- Main App Execution ---

st.title("SilentSignals: Real-Time Event Based Alerts")

# 1. Initialize session state
init_session_state()
solo_model = load_model()

if solo_model:
    # 2. Process any messages from the background thread
    process_alert_queue()

    # 3. Draw the main controls
    draw_control_panel(solo_model)
    draw_status_bar()

    # 4. Handle cleanup logic if "Stop" was just clicked
    if not st.session_state[STATE_RUNNING] and st.session_state[STATE_THREAD] is not None:
        with st.spinner("Stopping..."):
            # Wait for the thread to fully exit
            thread: Optional[threading.Thread] = st.session_state[STATE_THREAD]
            if thread:
                thread.join()
            st.session_state[STATE_THREAD] = None

            # Process any final messages left in the queue
            process_alert_queue()
            st.rerun() # Rerun one last time to draw the final "Stopped" state

    # 5. Draw the main UI sections
    st.markdown("---")
    st.subheader("Current Alerts")
    draw_active_alerts()
    
    st.subheader("Alert History")
    draw_tiered_alert_history()
    
    # 6. Trigger the live update loop (if running)
    run_live_update()
else:
    st.error("Model failed to load. The application cannot start.")
    st.warning("Please check your `config.yaml` file and model paths.")