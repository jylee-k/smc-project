import streamlit as st
import json
import time
import os
import threading
import queue
import numpy as np
import librosa
import soundfile as sf
from record_sound import ChunkRecorder
from realtime_solo import RealTimeSolo
import uuid
import io
import datetime # NEW: For history timestamp

# --- Page Config ---
st.set_page_config(
    page_title="SilentSignals",
    page_icon="🚨",
    layout="wide",
)

st.title("SilentSignals: Real-Time Event Based Alerts")

# --- Best Practice: Centralized State Management ---
def init_session_state():
    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False
    if 'pipeline_thread' not in st.session_state:
        st.session_state.pipeline_thread = None
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = None
    
    if 'alerts' not in st.session_state:
        st.session_state.alerts = {} 
        
    if 'alert_queue' not in st.session_state:
        st.session_state.alert_queue = queue.Queue()
        
    if 'audio_data_to_process' not in st.session_state:
        st.session_state.audio_data_to_process = None

    # NEW: For "Live" Audio Level meter
    if 'current_audio_level' not in st.session_state:
        st.session_state.current_audio_level = 0.0
        
    # NEW: For File Processing progress bar
    if 'file_progress' not in st.session_state:
        st.session_state.file_progress = 0.0
        
    # NEW: For Alert History
    if 'alert_history' not in st.session_state:
        st.session_state.alert_history = []

init_session_state()


# --- Background Tasks ---

def mic_pipeline(stop_event, alert_queue):
    """Handles microphone recording and processing."""
    try:
        solo = RealTimeSolo("./configs/config.yaml")
        solo.start_session()

        def handle_chunk(chunk):
            # NEW: Calculate audio level (RMS) and put on queue
            # We use a small epsilon to avoid log(0) errors
            rms = np.sqrt(np.mean(chunk**2) + 1e-10)
            # Normalize to a 0-1 range (approximate)
            level = float(np.clip(rms * 10, 0, 1))
            alert_queue.put({"level": level})

            result = solo.process_chunk(chunk, chunk_sr=solo.sr)
            if result:
                alert_queue.put(result)

        rec = ChunkRecorder(sr=solo.sr, channels=1,
                            chunk_size_sec=solo.chunk_ms / 1000.0,
                            on_chunk=handle_chunk)
        rec.start()
        stop_event.wait()
        rec.stop()
        solo.stop_session()
    except Exception as e:
        alert_queue.put({"error": f"Microphone Pipeline Error: {e}"})


def file_pipeline(stop_event, alert_queue, audio_bytes):
    """Handles audio file processing."""
    try:
        solo = RealTimeSolo("./configs/config.yaml")
        solo.start_session()
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            total_samples = len(y)
        except Exception as e:
            alert_queue.put({"error": f"Could not process audio file: {e}"})
            solo.stop_session()
            return

        hop = int(solo.sr * (solo.chunk_ms / 1000.0))
        for i in range(0, total_samples, hop):
            if stop_event.is_set():
                break
                
            # NEW: Calculate and send progress
            progress_percent = (i / total_samples)
            alert_queue.put({"progress": progress_percent})
            
            chunk = y[i:i + hop]
            if len(chunk) < hop:
                chunk = np.pad(chunk, (0, hop - len(chunk)))
            result = solo.process_chunk(chunk, chunk_sr=solo.sr)
            if result:
                alert_queue.put(result)
                
        alert_queue.put({"progress": 1.0}) # NEW: Signal completion
        solo.stop_session()
    except Exception as e:
        alert_queue.put({"error": f"File Processing Pipeline Error: {e}"})


# --- UI Layout ---
mode = st.radio("Audio Source", ["Microphone", "Audio File"], index=0, horizontal=True)

if mode == "Microphone":
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🎤 Start Recording", disabled=st.session_state.pipeline_running):
            st.session_state.alerts.clear()
            st.session_state.alert_history.clear()
            
            stop_event = threading.Event()
            thread = threading.Thread(target=mic_pipeline, args=(stop_event, st.session_state.alert_queue), daemon=True)
            thread.start()

            st.session_state.stop_event = stop_event
            st.session_state.pipeline_thread = thread
            st.session_state.pipeline_running = True
            st.success("Recording started!")
            st.rerun()

    with col2:
        if st.button("❌ Stop Recording", disabled=not st.session_state.pipeline_running):
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.session_state.pipeline_running = False
            st.session_state.current_audio_level = 0.0 # NEW: Reset level meter
            st.success("Recording stopped.")
            st.rerun()

    with col3:
        if st.button("🗑️ Clear All Alerts"):
            st.session_state.alerts.clear()
            st.session_state.alert_history.clear()
            st.rerun()

else:  # Audio File mode
    file_col, run_col, reset_col = st.columns([2, 1, 1])
    with file_col:
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
        
        if uploaded_file:
            st.session_state.audio_data_to_process = uploaded_file.read()
        elif 'audio_data_to_process' in st.session_state:
             st.session_state.audio_data_to_process = None


    with run_col:
        if st.button("▶️ Process File", disabled=(st.session_state.audio_data_to_process is None or st.session_state.pipeline_running)):
            st.session_state.alerts.clear()
            st.session_state.alert_history.clear()
            st.session_state.file_progress = 0.0 # NEW: Reset progress bar
            
            stop_event = threading.Event()
            audio_data = st.session_state.audio_data_to_process
            thread = threading.Thread(target=file_pipeline, args=(stop_event, st.session_state.alert_queue, audio_data), daemon=True)
            thread.start()

            st.session_state.stop_event = stop_event
            st.session_state.pipeline_thread = thread
            st.session_state.pipeline_running = True
            st.success("File processing started!")
            st.rerun()

    with reset_col:
        if st.button("🗑️ Clear All Alerts"):
            st.session_state.alerts.clear()
            st.session_state.alert_history.clear()
            st.rerun()

# --- Best Practice: Non-Blocking UI Updates & Alert Grouping ---
while not st.session_state.alert_queue.empty():
    item = st.session_state.alert_queue.get_nowait()
    
    if "error" in item:
        st.error(item["error"])
        st.session_state.pipeline_running = False
    
    # NEW: Handle audio level updates
    elif "level" in item:
        st.session_state.current_audio_level = item["level"]
        
    # NEW: Handle progress updates
    elif "progress" in item:
        st.session_state.file_progress = item["progress"]
    
    # Handle alert updates
    elif "message" in item:
        alert = item
        alert_key = alert.get("YAMNet", {}).get("top_label", "Unknown Event")

        if alert_key in st.session_state.alerts:
            st.session_state.alerts[alert_key]['count'] += 1
            st.session_state.alerts[alert_key]['alert_obj'] = alert
        else:
            # NEW: Show toast for new high-priority alerts
            tier = alert.get("tier", 1)
            if tier >= 2:
                icon = "🚨" if tier == 3 else "⚠️"
                st.toast(f"New Alert: {alert.get('message', '..._')}", icon=icon)

            st.session_state.alerts[alert_key] = {
                'alert_obj': alert,
                'count': 1,
                'first_seen': time.time(),
                'id': str(uuid.uuid4())
            }

# --- Status and Alert Display ---

# NEW: Create a status container
with st.container(border=True):
    if st.session_state.pipeline_running:
        if st.session_state.pipeline_thread and st.session_state.pipeline_thread.is_alive():
            st.markdown("**(Status: 🟢 Running)**")
        else:
            st.session_state.pipeline_running = False
            st.markdown("**(Status: 🏁 Finished)**")
            st.session_state.file_progress = 0.0 # Reset progress
            st.session_state.current_audio_level = 0.0 # Reset level
    else:
        st.markdown("**(Status: 🔴 Stopped)**")
        st.session_state.file_progress = 0.0
        st.session_state.current_audio_level = 0.0

    # NEW: Show Live Audio Level for Microphone mode
    if mode == "Microphone":
        st.progress(st.session_state.current_audio_level, text="Live Audio Level")

    # NEW: Show Progress Bar for File mode (only when running)
    if mode == "Audio File" and st.session_state.pipeline_running:
        st.progress(st.session_state.file_progress, text="File Processing Progress")


st.markdown("---")
st.subheader("Active Alerts:")

if not st.session_state.alerts:
    st.write("✅ No active alerts at the moment.")
else:
    sorted_alerts = sorted(st.session_state.alerts.items(), key=lambda item: item[1]['first_seen'], reverse=True)
    
    for alert_key, alert_data in sorted_alerts:
        
        # NEW: Use container for better visual grouping
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
            
            with col2:
                # NEW: Acknowledge button now moves to history
                if st.button("Acknowledge", key=alert_id):
                    timestamp = datetime.datetime.now().strftime("%I:%M:%S %p")
                    # Add to history
                    st.session_state.alert_history.insert(0, {
                        "text": alert_text,
                        "timestamp": timestamp,
                        "tier": tier
                    })
                    # Remove from active
                    del st.session_state.alerts[alert_key]
                    st.rerun()

# NEW: Show Alert History in an expander
if st.session_state.alert_history:
    st.markdown("---")
    with st.expander("Acknowledged Alert History"):
        for item in st.session_state.alert_history:
            st.markdown(f"*{item['timestamp']}* - {item['text']}")

if st.session_state.pipeline_running:
    # Rerun the page every 100ms to update the UI
    time.sleep(0.1) 
    st.rerun()