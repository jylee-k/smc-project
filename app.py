import streamlit as st
import json
import time
import os
import threading
import yaml
from datetime import datetime
import librosa
import numpy as np

#other scripts
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
os.makedirs(os.path.dirname(LOG_FILE_PATH) or ".", exist_ok=True)


# --- 1. MODEL CACHING ---
@st.cache_resource
def load_pipeline():
    """
    Loads the RealTimeSolo ML pipeline using Streamlit's caching.
    This ensures the model is loaded only once per session.
    """
    print("--- LOADING ML PIPELINE ---")
    try:
        pipeline = RealTimeSolo(CONFIG_FILE_PATH)
        print("--- ML Pipeline Loaded Successfully ---")
        return pipeline
    except Exception as e:
        print(f"ERROR: Could not load ML pipeline: {e}")
        st.error(f"Error loading ML pipeline: {e}. Please check config.yaml and model paths.")
        return None

solo_pipeline = load_pipeline()


# --- 2. BACKGROUND THREAD TARGET ---
def pipeline_target(stop_event_flag, pipeline_instance):
    """
    The main function executed in a separate thread for real-time audio processing.
    It sets up the ChunkRecorder and processes audio chunks via the pipeline.

    Args:
        stop_event_flag (threading.Event): An event to signal when the thread should stop.
        pipeline_instance (RealTimeSolo): The loaded ML pipeline instance.
    """
    solo = pipeline_instance
    if solo is None:
        print("Error: Pipeline instance is None. Thread cannot start.")
        st.error("ML Models not loaded. Cannot start pipeline.")
        return

    try:
        def on_chunk_callback(chunk):
            """Callback function passed to ChunkRecorder."""
            try:
                if not stop_event_flag.is_set():
                    solo.process_chunk(chunk, chunk_sr=16000)
                else:
                    print("Stop event set, skipping chunk processing.")
            except Exception as e:
                print(f"Error processing chunk: {e}")

        recorder = ChunkRecorder(
            sr=16000,
            channels=1,
            chunk_size_sec=2.0, #change this for chunk size change from microphone
            on_chunk=on_chunk_callback
        )
        
        print("Starting pipeline session...")
        solo.start_session()
        recorder.start()

        while not stop_event_flag.is_set():
            time.sleep(0.1)

        print("Stopping pipeline session...")
        recorder.stop()
        solo.stop_session()
        print("Pipeline thread stopped without issues.")
    except Exception as e:
        print(f"Error in pipeline thread: {e}")


# --- 3. SESSION STATE ---
def initialize_session_state():
    """
    Sets up the initial Streamlit session state variables.
    This runs once at the beginning of the session.
    """
    if not os.path.exists(LOG_FILE_PATH):
        open(LOG_FILE_PATH, 'w').close()
    start_pos = 0
    try:
        with open(LOG_FILE_PATH, 'r') as f:
            f.seek(0, os.SEEK_END)
            start_pos = f.tell()
    except Exception as e:
        print(f"Warning: Could not get log file size. {e}")
        start_pos = 0

    ALL_LABELS = []
    if solo_pipeline and hasattr(solo_pipeline, 'custom_labels') and solo_pipeline.custom_labels:
        ALL_LABELS = sorted(solo_pipeline.custom_labels)
    else:
        print("Warning: Could not load custom_labels from pipeline for personalization.")

    # Define default tiers based on keywords
    # --- Tier 3: Critical (Immediate Danger / High Priority) ---
    _critical_substrings = [
        "fire alarm", "smoke detector", "siren", "screaming", "baby cry", 
        "explosion", "gunshot", "machine gun", "breaking", "shatter", "glass"
    ]
    
    # --- Tier 2: Warning (Needs Attention) ---
    _warning_substrings = [
        "car alarm", "alarm clock", "shout", "crying", "slam", "ringing", 
        "ringtone", "horn", "bark", "dog", "thunder", "fireworks", "firecracker"
    ]

    # --- Tier 1: Info (Contextual / Environmental) ---
    _info_substrings = [
        "doorbell", "knock", "footsteps", "door", "keys jangling", "typing", 
        "keyboard", "dishes", "cutlery", "chopping", "frying", "microwave", 
        "blender", "water", "sink", "bathtub", "toilet flush", "hair dryer", 
        "vacuum cleaner", "car passing by", "bus", "truck", "motorcycle", 
        "train", "subway", "aircraft", "helicopter", "bicycle", "skateboard", 
        "rain", "wind", "bird", "cat", "meow", "applause", "laughter", 
        "conversation", "speech", "vehicle"
    ]

    # --- Automatically build the default tier lists from the keywords ---
    DEFAULT_CRITICAL = [lbl for lbl in ALL_LABELS if any(s in lbl.lower() for s in _critical_substrings)]
    DEFAULT_WARNING = [lbl for lbl in ALL_LABELS if any(s in lbl.lower() for s in _warning_substrings)]
    DEFAULT_INFO = [lbl for lbl in ALL_LABELS if any(s in lbl.lower() for s in _info_substrings)]

    defaults = {
        'log_file': LOG_FILE_PATH,
        'pipeline_running': False,
        'pipeline_thread': None,
        'stop_event': None,
        'file_pos': start_pos,
        'alerts': [],
        'acknowledged_ids': set(),
        'last_prediction': None,
        'active_profile': 'Normal',
        'all_labels': ALL_LABELS,
        'critical_tier_labels': DEFAULT_CRITICAL,
        'warning_tier_labels': DEFAULT_WARNING,
        'info_tier_labels': DEFAULT_INFO,
        'critical_threshold': 0.6,
        'warning_threshold': 0.6,
        'info_threshold': 0.6,
        'processing_file': False,
        'last_file_id': None,
        'critical_dialog_open': False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()
LOG_FILE = st.session_state.log_file


# --- 4. HELPER FUNCTIONS ---

@st.dialog("🚨 Critical Alert Detected!", dismissible=False)
def show_critical_alert_dialog(alert):
    """
    Displays a modal dialog for critical (Tier 3) alerts.
    This function is called on *every* rerun as long as the alert is active.
    """
    st.session_state.critical_dialog_open = True
    
    st.markdown(f"## {alert['type']}")
    st.markdown(f"**Score:** {alert['message']}")
    try:
        alert_time = datetime.fromtimestamp(alert['time']).strftime('%I:%M:%S %p')
    except Exception:
        alert_time = "Just now"
    st.markdown(f"**Time:** {alert_time}")
    st.warning("This is a critical alert and requires your attention.")

    if st.button("Acknowledge", use_container_width=True, type="primary"):
        # 1. Acknowledge the alert
        new_ids = st.session_state.acknowledged_ids.copy()
        new_ids.add(alert['id'])
        st.session_state.acknowledged_ids = new_ids
        
        # 2. Stop the microphone pipeline if it's running
        if st.session_state.pipeline_running:
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.session_state.pipeline_running = False
            st.success("Alert acknowledgment, stop microphone.")
            time.sleep(0.5) 
        
        # 3. Close the dialog and rerun
        st.session_state.critical_dialog_open = False
        st.rerun()

def parse_prediction_to_alert(alert_data):
    """
    Parses a raw prediction dictionary from the log file into a formatted alert object.
    Applies tier logic, confidence thresholds, and profile (DND/Sleep) filters.
    """
    pred = alert_data.get("Fused")
    if not pred:
        return None

    label = pred.get("top_label", "Unknown")
    score = pred.get("top_score", 0)
    tier = 0
    message = f"{score:.1%}"
    label_lower = label.lower()

    if label_lower in [l.lower() for l in st.session_state.critical_tier_labels] and score > st.session_state.critical_threshold:
        tier = 3
    elif label_lower in [l.lower() for l in st.session_state.warning_tier_labels] and score > st.session_state.warning_threshold:
        tier = 2
    elif label_lower in [l.lower() for l in st.session_state.info_tier_labels] and score > st.session_state.info_threshold:
        tier = 1

    active_profile = st.session_state.active_profile
    if active_profile == 'DND' and tier < 3:
        return None
    if active_profile == 'Sleep' and tier < 2:
        return None

    if tier > 0:
        group_id = f"{tier}-{label.capitalize()}"
        alert_time = alert_data.get('time_start', time.time())
        session_start_time = solo_pipeline.session_t0 if (solo_pipeline and solo_pipeline.session_t0) else time.time()
        
        return {
            "id": group_id, "tier": tier, "type": label.capitalize(),
            "message": message, "time": session_start_time + alert_time
        }
    return None

def display_alert_card(alert):
    """
    Renders a single alert card in the UI.
    This function handles the visual state for acknowledged vs. unacknowledged alerts.
    """
    icon = "ℹ️"
    if alert['tier'] == 3: 
        icon = "🚨"
    elif alert['tier'] == 2: 
        icon = "⚠️"

    is_acknowledged = alert['id'] in st.session_state.acknowledged_ids

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 6, 2], vertical_alignment="center")

        with c1:
            if is_acknowledged:
                st.markdown(f"<span style='font-size: 32px;'>✅</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='font-size: 32px;'>{icon}</span>", unsafe_allow_html=True)

        with c2:
            try:
                alert_time = datetime.fromtimestamp(alert['time']).strftime('%I:%M:%S %p')
            except Exception:
                alert_time = "Just now"

            if is_acknowledged:
                st.markdown(f"**{alert['type']}** detected (Acknowledged)")
            else:
                st.markdown(f"**{alert['type']}** detected")

            st.caption(f"Score: {alert['message']} | {alert_time}")

        with c3:
            if alert['tier'] >= 2:
                if st.button("Acknowledge", key=f"ack_{alert['id']}",
                             help="Acknowledge this alert",
                             disabled=is_acknowledged):
                    
                    st.session_state.acknowledged_ids.add(alert['id'])
                    st.rerun()


# --- 5. SIDEBAR CONTROLS ---
st.sidebar.title("Microphone Status")
if st.session_state.pipeline_running:
    st.sidebar.markdown("**Microphone Status:** 🟢 Running")
else:
    st.sidebar.markdown("**Microphone Status:** 🔴 Stopped")
if solo_pipeline is None:
    st.sidebar.error("Models failed to load. Cannot start.")
st.sidebar.divider()

st.sidebar.radio(
    "Active Profile", ["Normal", "Sleep", "DND"],
    key='active_profile', horizontal=False,
)
st.sidebar.caption("**Normal**: All. **Sleep**: Warning & Critical. **DND**: Critical only.")

if st.sidebar.button("🎤 Start Microphone", use_container_width=True, disabled=(solo_pipeline is None)):
    if not st.session_state.pipeline_running:
        stop_event = threading.Event()
        thread = threading.Thread(target=pipeline_target, args=(stop_event, solo_pipeline,), daemon=True)
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
    st.session_state.last_prediction = None
    st.warning("Alerts reset")
    st.rerun()
st.sidebar.divider()

with st.sidebar.expander("Customize Alert Tiers"):
    if not st.session_state.all_labels:
        st.warning("Model labels not loaded. Cannot customize tiers.")
    else:
        st.multiselect("🚨 Tier 3: Critical", options=st.session_state.all_labels, key="critical_tier_labels")
        st.multiselect("⚠️ Tier 2: Warning", options=st.session_state.all_labels, key="warning_tier_labels")
        st.multiselect("ℹ️ Tier 1: Info", options=st.session_state.all_labels, key="info_tier_labels")
        st.divider()
        st.slider("🚨 Critical Confidence", 0.0, 1.0, key="critical_threshold", step=0.05)
        st.slider("⚠️ Warning Confidence", 0.0, 1.0, key="warning_threshold", step=0.05)
        st.slider("ℹ️ Info Confidence", 0.0, 1.0, key="info_threshold", step=0.05)

st.sidebar.divider()
st.sidebar.markdown("### 🔈 Demo Mode")
uploaded_file = st.sidebar.file_uploader(
    "Test with an audio file", type=["wav", "mp3", "flac"],
    help="Uploading a file will stop the microphone and process the file instead."
)


# --- 6. MAIN PAGE ---
st.title("SilentSignals: Real-Time Event Based Alerts")

tab_dashboard, tab_finetune = st.tabs(["🚨 Live Dashboard", "🔬 Submit for Finetuning"])

with tab_dashboard:
    if uploaded_file is not None:
        file_id = (uploaded_file.name, uploaded_file.size)
        if file_id != st.session_state.get('last_file_id'):
            st.session_state.processing_file = True
            
            if st.session_state.pipeline_running:
                if st.session_state.stop_event:
                    st.session_state.stop_event.set()
                st.session_state.pipeline_running = False
                st.warning("Microphone stopped to process uploaded file.")
                time.sleep(0.5)

            st.info(f"Processing uploaded file: `{uploaded_file.name}`. Please wait...")

            try:
                y, sr = librosa.load(uploaded_file, sr=16000, mono=True)
                chunk_samples = int(16000 * 2.0)
                
                if solo_pipeline:
                    solo_pipeline.start_session()
                    progress_bar = st.progress(0, "Processing audio...")
                    total_chunks = (len(y) + chunk_samples - 1) // chunk_samples

                    for i, chunk_num in enumerate(range(0, len(y), chunk_samples)):
                        chunk = y[chunk_num : chunk_num + chunk_samples]
                        if len(chunk) < chunk_samples:
                            pad = np.zeros(chunk_samples - len(chunk), dtype=np.float32)
                            chunk = np.concatenate([chunk, pad])
                        
                        solo_pipeline.process_chunk(chunk, chunk_sr=16000)
                        progress_bar.progress((i + 1) / total_chunks, f"Processing chunk {i+1}/{total_chunks}")
                        
                    solo_pipeline.stop_session()
                    progress_bar.empty()
                    st.success(f"File processing complete. {total_chunks} chunks analyzed.")
                    st.session_state.last_file_id = file_id
                else:
                    st.error("Models not loaded. Cannot process file.")
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                st.session_state.processing_file = False
                st.rerun()

    col_alerts, col_live = st.columns([1, 1])

    with col_live:
        st.subheader("📈 Live Detections")
        live_container = st.container(border=True)
        
        if st.session_state.processing_file:
            live_container.info("Demo file processing in progress...")
        elif st.session_state.last_prediction is None:
            live_container.info("Start the microphone or upload a file.")
        else:
            pred_data = st.session_state.last_prediction
            fused_pred = pred_data.get("Fused", {})
            pann_pred = pred_data.get("PANN", {})
            vgg_pred = pred_data.get("VGGish", {})
            ast_pred = pred_data.get("AST", {})

            fused_label = fused_pred.get("top_label", "N/A")
            fused_score = fused_pred.get("top_score", 0)
            
            #say no output if below confidence score
            if fused_score < 0.60: 
                fused_label = "No output"
            
            live_container.metric("**Fused Model**", fused_label, f"{fused_score:.1%} Confidence")
            
            live_container.divider()
            m_col1, m_col2, m_col3 = live_container.columns(3)
            with m_col1:
                st.markdown("**PANN**")
                pann_label = pann_pred.get("top_label", "N/A")
                pann_score = pann_pred.get("top_score", 0)
                st.write(f"{pann_label}")
                st.caption(f"({pann_score:.1%} Confidence)")
            with m_col2:
                st.markdown("**VGGish**")
                vgg_label = vgg_pred.get("top_label", "N/A")
                vgg_score = vgg_pred.get("top_score", 0)
                st.write(f"{vgg_label}")
                st.caption(f"({vgg_score:.1%} Confidence)")
            with m_col3:
                st.markdown("**AST**")
                ast_label = ast_pred.get("top_label", "N/A")
                ast_score = ast_pred.get("top_score", 0)
                st.write(f"{ast_label}")
                st.caption(f"({ast_score:.1%} Confidence)")

    with col_alerts:
        # --- 7. ALERT PROCESSING ---
        try:
            with open(LOG_FILE, 'r') as f:
                f.seek(st.session_state.file_pos)
                new_lines = f.readlines()
                st.session_state.file_pos = f.tell()
        except FileNotFoundError:
            new_lines = []

        newly_created_alerts = []
        for line in new_lines:
            try:
                alert_data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not st.session_state.processing_file:
                st.session_state.last_prediction = alert_data
                
            new_alert = parse_prediction_to_alert(alert_data)
            
            if new_alert:
                group_id = new_alert['id']
                found_index = -1
                for i, alert in enumerate(st.session_state.alerts):
                    if alert['id'] == group_id:
                        found_index = i
                        break
                
                if found_index != -1:
                    st.session_state.alerts[found_index] = new_alert
                else:
                    st.session_state.alerts.append(new_alert)
                
                newly_created_alerts.append(new_alert)

        # --- 7a. Show Persistent Critical Dialog ---
        alert_to_show = None
        for alert in st.session_state.alerts:
            if alert['tier'] == 3 and alert['id'] not in st.session_state.acknowledged_ids:
                alert_to_show = alert
                break

        if alert_to_show and not st.session_state.processing_file:
            show_critical_alert_dialog(alert_to_show)

        # --- 7b. Show Toasts for *new* non-critical alerts ---
        newly_created_alerts.sort(key=lambda x: x.get('tier', 0))
        for alert in newly_created_alerts:
            if alert['id'] in st.session_state.acknowledged_ids or alert['tier'] == 3:
                continue

            if alert['tier'] == 2:
                st.toast(f"⚠️ Warning: {alert['type']} detected!", icon="⚠️")
        
        # --- 8. TIERED ALERT DISPLAY ---
        all_active = st.session_state.alerts
        all_active.sort(key=lambda x: x.get('time', 0), reverse=True)
        st.subheader(f"🚨 Active Alerts ({len(all_active)})")

        tier3_alerts = [a for a in all_active if a.get('tier') == 3]
        tier2_alerts = [a for a in all_active if a.get('tier') == 2]
        tier1_alerts = [a for a in all_active if a.get('tier') == 1]
        active_profile = st.session_state.active_profile

        with st.expander(f"🚨 Tier 3: Critical ({len(tier3_alerts)})", expanded=True):
            st.caption("📱 **Action:** Strong-Vibration, Phone Camera Flash & Smartwatch Alert")
            if not tier3_alerts: 
                st.write("✅ No critical alerts.")
            else:
                for alert in tier3_alerts: 
                    display_alert_card(alert)
        
        if active_profile in ['Normal', 'Sleep']:
            with st.expander(f"⚠️ Tier 2: Warning ({len(tier2_alerts)})", expanded=True):
                st.caption("⌚️ **Action:** Phone Soft-Vibration & Smartwatch Alert")
                if not tier2_alerts: 
                    st.write("✅ No warning alerts.")
                else:
                    for alert in tier2_alerts: 
                        display_alert_card(alert)
        
        if active_profile == 'Normal':
            with st.expander(f"ℹ️ Tier 1: Info ({len(tier1_alerts)})", expanded=True):
                st.caption("ℹ️ **Action:** Standard Phone Notification")
                if not tier1_alerts:
                    st.write("✅ No info alerts.")
                else:
                    for alert in tier1_alerts: 
                        display_alert_card(alert)

with tab_finetune:
    st.subheader("Submit Audio for Model Improvement")
    st.info("Have an audio clip that wasn't detected correctly? You can help us improve the model by submitting it here with the correct label.")
    
    with st.form("finetune_form"):
        new_audio_file = st.file_uploader(
            "Upload Audio File",
            type=["wav", "mp3", "flac"]
        )
        new_label = st.text_input(
            "Enter Correct Label",
            placeholder="e.g., 'Dog barking', 'Specific type of alarm'"
        )
        submitted = st.form_submit_button("Submit for Processing")
        
        if submitted:
            if not new_audio_file or not new_label.strip():
                st.error("Please provide both an audio file and a label.")
            else:
                st.success("Thank you! The audio clip will proceed to finetune with the new labels.")

# --- 9. AUTO-REFRESH LOOP ---
if st.session_state.pipeline_running and not st.session_state.get('critical_dialog_open', False):
    try:
        time.sleep(1)
        st.rerun()
    except Exception as e:
        pass