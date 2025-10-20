import streamlit as st
import json, time, os, threading, io
import numpy as np
import librosa
import soundfile as sf
from record_sound import ChunkRecorder
from realtime_solo import RealTimeSolo

st.set_page_config(
    page_title="SilentSignals",
    page_icon="🚨",
    layout="wide",
    )

st.title("SilentSignals: Real-Time Event Based Alerts")

# Initialize session state variables 
if 'log_file' not in st.session_state:
    st.session_state.log_file = "./runs/stream_preds_sample.jsonl"
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'pipeline_thread' not in st.session_state:
    st.session_state.pipeline_thread = None
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'file_pos' not in st.session_state:
    st.session_state.file_pos = 0
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'acknowledged_ids' not in st.session_state:
    st.session_state.acknowledged_ids = set()

LOG_FILE = st.session_state.log_file

# Ensure the log file exists
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, 'w').close()


mode = st.radio("Audio Source", ["Microphone", "Audio File"], index=0)
if mode == "Microphone":
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("🎤 Start Recording"):
            if not st.session_state.pipeline_running:
                # Prepare log for new session
                open(st.session_state.log_file, 'w').close()
                st.session_state.file_pos = 0
                # Start background thread for mic
                stop_event = threading.Event()
                def run_pipeline(stop_event):
                    solo = RealTimeSolo("./configs/config.yaml")
                    solo.start_session()
                    rec = ChunkRecorder(sr=solo.sr, channels=1,
                                        chunk_size_sec=solo.chunk_ms/1000.0,
                                        on_chunk=lambda chunk: solo.process_chunk(chunk, chunk_sr=solo.sr))
                    rec.start()
                    # wait until stop signal
                    while not stop_event.is_set():
                        time.sleep(0.1)
                    rec.stop()
                    solo.stop_session()
                thread = threading.Thread(target=run_pipeline, args=(stop_event,), daemon=True)
                thread.start()
                st.session_state.stop_event = stop_event
                st.session_state.pipeline_thread = thread
                st.session_state.pipeline_running = True
                st.success("Pipeline started")
            else:
                st.warning("Pipeline is already running.")
    with col2:
        if st.button("❌ Stop Recording"):
            if st.session_state.pipeline_running:
                st.session_state.stop_event.set()
                # Optionally, join thread here
                st.session_state.pipeline_running = False
                st.success("Pipeline stopped")
            else:
                st.info("No recording in progress.")
    with col3:
        if st.button("Reset Alerts"):
            open(st.session_state.log_file, 'w').close()
            st.session_state.alerts = []
            st.session_state.acknowledged_ids = set()
            st.session_state.file_pos = 0
            st.warning("Alerts reset")
else:  # Audio File mode
    file_col, run_col, reset_col = st.columns([2,1,1])
    with file_col:
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav","mp3","ogg"])
    with run_col:
        if st.button("▶️ Process File"):
            if uploaded_file is not None:
                if not st.session_state.pipeline_running:
                    # Clear log and start file processing thread
                    open(st.session_state.log_file, 'w').close()
                    st.session_state.file_pos = 0
                    data = uploaded_file.read()
                    stop_event = threading.Event()
                    def run_pipeline_file(stop_event, audio_bytes):
                        solo = RealTimeSolo("./configs/config.yaml")
                        solo.start_session()
                        # Load audio from bytes
                        try:
                            import io
                            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
                        except Exception as e:
                            print("Error loading file:", e)
                            try:
                                # fallback to soundfile
                                y, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
                                if sr != 16000:
                                    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
                                    sr = 16000
                                if y.ndim > 1:
                                    y = np.mean(y, axis=1)
                            except Exception as e2:
                                print("Could not process audio file.", e2)
                                solo.stop_session()
                                return
                        
                        hop = int(solo.sr * (solo.chunk_ms/1000.0))
                        for i in range(0, len(y), hop):
                            if stop_event.is_set():
                                break
                            chunk = y[i : i+hop]
                            if len(chunk) < hop:
                                chunk = np.pad(chunk, (0, hop - len(chunk)))
                            solo.process_chunk(chunk, chunk_sr=solo.sr)
                            # (Optional: time.sleep(chunk_duration) to simulate real time)
                        solo.stop_session()
                    thread = threading.Thread(target=run_pipeline_file, args=(stop_event, data), daemon=True)
                    thread.start()
                    st.session_state.stop_event = stop_event
                    st.session_state.pipeline_thread = thread
                    st.session_state.pipeline_running = True
                    st.success("Processing started")
                else:
                    st.warning("Another pipeline is running. Please stop it first.")
            else:
                st.error("Please upload a file first.")
    with reset_col:
        if st.button("Reset Alerts"):
            open(st.session_state.log_file, 'w').close()
            st.session_state.alerts = []
            st.session_state.acknowledged_ids = set()
            st.session_state.file_pos = 0
            st.warning("Alerts reset")


# Show pipeline status
if st.session_state.pipeline_thread:
    if not st.session_state.pipeline_thread.is_alive() and st.session_state.pipeline_running:
        # background thread finished (for file input)
        st.session_state.pipeline_running = False
        st.success("Pipeline finished")
        
if st.session_state.pipeline_running:
    st.markdown("**Microphone Status:** 🟢 Running")
else:
    st.markdown("**Microphone Status:** 🔴 Stopped")

st.markdown("---")

# --- Alert Display Section ---
st.subheader("Active Alerts:")

with open(LOG_FILE, 'r') as f:
    f.seek(st.session_state.file_pos)
    new_lines = f.readlines()
    st.session_state.file_pos = f.tell()
    
for line in new_lines:
    try:
        alert = json.loads(line)
    except json.JSONDecodeError:
        continue
    alert_id = alert.get("id", None) or line
    if alert_id in st.session_state.acknowledged_ids:
        continue  # skip alerts already acknowledged
    st.session_state.alerts.append(alert) # Add new alert to list

# Display alerts cards for each active alert
for alert in list(st.session_state.alerts):
    alert_id = alert.get("id", None) or alert
    if alert_id in st.session_state.acknowledged_ids:
        continue  # already acknowledged, skip
    tier = alert.get("tier", 1)
    alert_type = alert.get("type", "Alert")
    location = alert.get("location", "Unknown location")
    message = alert.get("message", "")
    # Compose alert text
    alert_text = f"**{alert_type} Alert:** {message}"
    if location:
        alert_text += f" at *{location}*"
    # Display with tier-based styling
    if tier >= 3:
        st.error(alert_text, icon="🚨")
    elif tier == 2:
        st.warning(alert_text, icon="⚠️")
    else:
        st.info(alert_text, icon="ℹ️")
    
    # Device indicators (for critical alerts, show all; for others, maybe partial)
    device_line = "Devices: "
    if tier >= 3:
        device_line += "🔴 LED Flash, 📳 Phone, ⌚ Watch"
    elif tier == 2:
        device_line += "📳 Phone, ⌚ Watch"
    else:
        device_line += "⌚ Watch"  # or no devices for tier1, adjust as needed
    st.write(device_line)
    
    # Acknowledge button
    ack_col = st.columns(3)[1]
    if ack_col.button("Acknowledge", key=str(alert_id)):
        st.session_state.acknowledged_ids.add(alert_id)
        # Remove from active alerts list
        st.session_state.alerts.remove(alert)
        st.experimental_rerun()  # refresh to update the list immediately

# If no alerts left, inform the user
if not st.session_state.alerts:
    st.write("✅ No active alerts at the moment.")
