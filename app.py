import streamlit as st
import json, time, os, threading

st.set_page_config(
    page_title="SilentSignals",
    page_icon="🚨",
    layout="wide",
    )

st.title("SilentSignals: Real-Time Event Based Alerts")

# Initialize session state variables 
if 'log_file' not in st.session_state:
    st.session_state.log_file = "./data/stream_preds_sample.jsonl"
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


col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("🎤 Start Microphone"):
        if not st.session_state.pipeline_running:
            stop_event = threading.Event()
            thread = threading.Thread(target=None, args=(stop_event,), daemon=True)
            thread.start()
            st.session_state.stop_event = stop_event
            st.session_state.pipeline_thread = thread
            st.session_state.pipeline_running = True
            st.success("Pipeline started")
with col2:
    if st.button("❌ Stop Microphone"):
        if st.session_state.pipeline_running:
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.session_state.pipeline_running = False
            st.success("Pipeline stopped")
with col3:
    if st.button("Reset Alerts"):
        # Clear log file and internal lists
        open(LOG_FILE, 'w').close()
        st.session_state.alerts = []
        st.session_state.acknowledged_ids = set()
        st.session_state.file_pos = 0
        st.warning("Alerts reset")

# Show pipeline status
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
