import streamlit as st
import json, time, os, threading
import yaml  # Not strictly needed, but good practice if config expands

# Import the necessary classes from your other files
from record_sound import ChunkRecorder
from realtime_solo import RealTimeSolo

st.set_page_config(
    page_title="SilentSignals",
    page_icon="🚨",
    layout="wide",
)

# --- CONFIGURATION ---
LOG_FILE_PATH = "runs/stream_preds.jsonl" 
CONFIG_FILE_PATH = "config.yaml"

# Ensure the 'runs' directory exists
os.makedirs(os.path.dirname(LOG_FILE_PATH) or ".", exist_ok=True)


# --- BACKGROUND PIPELINE THREAD ---
def pipeline_target(stop_event_flag):
    """
    Target function for the audio processing thread.
    This function will run in the background.
    """
    try:
        # 1. Initialize the ML pipeline
        solo = RealTimeSolo(CONFIG_FILE_PATH) 

        # 2. Define the callback for when audio chunks are ready
        def on_chunk_callback(chunk):
            """This callback passes audio chunks to the ML model."""
            try:
                if not stop_event_flag.is_set():
                    solo.process_chunk(chunk, chunk_sr=16000)
                else:
                    print("Stop event set, skipping chunk processing.")
            except Exception as e:
                print(f"Error processing chunk: {e}") # Log to console

        # 3. Initialize the recorder
        recorder = ChunkRecorder(
            sr=16000, 
            channels=1, 
            chunk_size_sec=2.0, 
            on_chunk=on_chunk_callback
        )

        # 4. Start processing
        print("Starting pipeline session...")
        solo.start_session()
        recorder.start()
        
        # 5. Wait for the stop signal
        while not stop_event_flag.is_set():
            time.sleep(0.1) # Polling to keep the thread alive

        # 6. Stop and clean up
        print("Stopping pipeline session...")
        recorder.stop()
        solo.stop_session()
        print("Pipeline thread stopped gracefully.")

    except Exception as e:
        print(f"Error in pipeline thread: {e}")


# --- SESSION STATE INITIALIZATION ---
if 'log_file' not in st.session_state:
    st.session_state.log_file = LOG_FILE_PATH
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
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None


LOG_FILE = st.session_state.log_file

# Ensure the log file exists
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, 'w').close()


# --- MODIFIED: SIDEBAR CONTROLS ---
st.sidebar.title("Controls")

if st.sidebar.button("🎤 Start Microphone", use_container_width=True):
    if not st.session_state.pipeline_running:
        stop_event = threading.Event()
        thread = threading.Thread(
            target=pipeline_target, 
            args=(stop_event,), 
            daemon=True
        )
        thread.start()
        st.session_state.stop_event = stop_event
        st.session_state.pipeline_thread = thread
        st.session_state.pipeline_running = True
        st.success("Pipeline started")
        st.rerun() 

if st.sidebar.button("❌ Stop Microphone", use_container_width=True):
    if st.session_state.pipeline_running:
        if st.session_state.stop_event:
            st.session_state.stop_event.set()
        st.session_state.pipeline_running = False
        time.sleep(0.5) 
        st.success("Pipeline stopped")
        st.rerun() 

if st.sidebar.button("🗑️ Reset Alerts", use_container_width=True):
    open(LOG_FILE, 'w').close()
    st.session_state.alerts = []
    st.session_state.acknowledged_ids = set()
    st.session_state.file_pos = 0
    st.session_state.last_prediction = None # Reset live view
    st.warning("Alerts reset")
    st.rerun() 

# Show pipeline status in sidebar
st.sidebar.divider()
if st.session_state.pipeline_running:
    st.sidebar.markdown("**Microphone Status:** 🟢 Running")
else:
    st.sidebar.markdown("**Microphone Status:** 🔴 Stopped")


# --- MAIN PAGE ---
st.title("SilentSignals: Real-Time Event Based Alerts")

# --- MODIFIED: LIVE MODEL DETECTION (in an Expander) ---
with st.expander("📈 Show Live Model Detections", expanded=False):
    live_container = st.container() # No border needed inside expander
    if st.session_state.last_prediction is None:
        live_container.info("Start the microphone to see live detections.")
    else:
        pred_data = st.session_state.last_prediction
        
        # Get data for each model, with fallbacks
        fused_pred = pred_data.get("Fused", {})
        pann_pred = pred_data.get("PANN", {})
        vgg_pred = pred_data.get("VGGish", {})
        ast_pred = pred_data.get("AST", {})

        # --- THIS IS THE FIRST FIX ---
        m_col1, m_col2, m_col3, m_col4 = live_container.columns(4)
        
        m_col1.metric(
            label="**Fused Model**", 
            value=fused_pred.get("top_label", "N/A"),
            delta=f"{fused_pred.get('top_score', 0):.2%} Conf."
        )
        m_col2.metric(
            label="PANN", 
            value=pann_pred.get("top_label", "N/A"),
            delta=f"{pann_pred.get('top_score', 0):.2%} Conf."
        )
        m_col3.metric(
            label="VGGish", 
            value=vgg_pred.get("top_label", "N/A"),
            delta=f"{vgg_pred.get('top_score', 0):.2%} Conf."
        )
        # --- THIS IS THE SECOND FIX ---
        m_col4.metric(
            label="AST", 
            value=ast_pred.get("top_label", "N/A"),
            delta=f"{ast_pred.get('top_score', 0):.2%} Conf."
        )


# --- ALERT PROCESSING SECTION (Reading from log) ---
try:
    with open(LOG_FILE, 'r') as f:
        f.seek(st.session_state.file_pos)
        new_lines = f.readlines()
        st.session_state.file_pos = f.tell()
except FileNotFoundError:
    new_lines = [] 
    
for line in new_lines:
    try:
        alert_data = json.loads(line)
    except json.JSONDecodeError:
        continue
    
    # Update live prediction state
    st.session_state.last_prediction = alert_data

    # --- Business Logic: Translate ML output to a UI alert ---
    pred = alert_data.get("Fused")
    if not pred:
        pred = alert_data.get("VGGish", alert_data.get("PANN", {}))

    label = pred.get("top_label", "Unknown")
    score = pred.get("top_score", 0)
    alert_id = f"{alert_data.get('time_start', time.time())}-{label}"
    
    if alert_id in st.session_state.acknowledged_ids:
        continue  
    
    # --- Customize Your Alert Rules Here ---
    tier = 0 # Default: 0 = No alert
    message = f"{label.capitalize()} detected (Score: {score:.2f})"
    
    critical_sounds = ["screaming", "baby cry", "cry", "glass break", "gunshot", "siren"]
    warning_sounds = ["dog bark", "cough", "snoring", "alarm", "smoke"]
    info_sounds = ["speech", "music", "typing"] # Example
    
    normalized_label = label.lower()
    
    if any(sound in normalized_label for sound in critical_sounds) and score > 0.5:
        tier = 3
    elif any(sound in normalized_label for sound in warning_sounds) and score > 0.4:
        tier = 2
    elif any(sound in normalized_label for sound in info_sounds) and score > 0.6:
        tier = 1
    
    # Only create an alert if tier is 1 or higher
    if tier > 0:
        new_alert = {
            "id": alert_id,
            "tier": tier,
            "type": label.capitalize(),
            "location": "Main Room", 
            "message": message,
            "time": alert_data.get('time_start', time.time()) # Store time for sorting
        }
        
        # Avoid adding duplicates
        if not any(a['id'] == new_alert['id'] for a in st.session_state.alerts):
             st.session_state.alerts.append(new_alert)


# --- MODIFIED: TIERED ALERT DISPLAY (in Tabs) ---
st.subheader("🚨 Active Alerts")

# Filter and sort alerts
all_active = [a for a in st.session_state.alerts if a['id'] not in st.session_state.acknowledged_ids]
all_active.sort(key=lambda x: x.get('time', 0), reverse=True) 

tier3_alerts = [a for a in all_active if a.get('tier') == 3]
tier2_alerts = [a for a in all_active if a.get('tier') == 2]
tier1_alerts = [a for a in all_active if a.get('tier') == 1]

# Create tabs
tab1, tab2, tab3 = st.tabs([
    f"🚨 Tier 3: Critical ({len(tier3_alerts)})", 
    f"⚠️ Tier 2: Warning ({len(tier2_alerts)})", 
    f"ℹ️ Tier 1: Info ({len(tier1_alerts)})"
])

# Helper function to display an alert card
def display_alert(alert):
    with st.container(border=True):
        alert_text = f"**{alert['type']} Alert:** {alert['message']} at *{alert['location']}*"
        st.markdown(alert_text)
        
        devices = "⌚ Watch" # Default
        if alert['tier'] == 2:
            devices = "📳 Phone, ⌚ Watch"
        elif alert['tier'] == 3:
            devices = "🔴 LED Flash, 📳 Phone, ⌚ Watch"
        st.write(f"Devices: {devices}")
        
        if st.button("Acknowledge", key=f"ack_{alert['id']}"):
            st.session_state.acknowledged_ids.add(alert['id'])
            st.rerun()

# Populate Tier 3 Tab
with tab1:
    if not tier3_alerts:
        st.write("✅ No critical alerts.")
    for alert in tier3_alerts:
        display_alert(alert)

# Populate Tier 2 Tab
with tab2:
    if not tier2_alerts:
        st.write("✅ No warning alerts.")
    for alert in tier2_alerts:
        display_alert(alert)

# Populate Tier 1 Tab
with tab3:
    if not tier1_alerts:
        st.write("✅ No info alerts.")
    for alert in tier1_alerts:
        display_alert(alert)


# --- AUTO-REFRESH LOOP ---
if st.session_state.pipeline_running:
    try:
        time.sleep(2) # Refresh interval in seconds
        st.rerun() 
    except Exception as e:
        pass