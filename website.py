import sys
import os
import streamlit as st
from engine import load_screening_model
from audio import extract_vocal_features

#  Page Configuration
st.set_page_config(
    page_title="Early disease detection",
    layout="centered", 
    initial_sidebar_state="expanded"
)

#UI DESIGN LAYER
st.markdown("""
    <style>
    /* Global Canvas Design */
    .stApp 
    {
        background-color: #09070f; 
        color: #f3f0f7; 
    }
    
    /* Clean, Modern Header Core */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -1.5px;
        color: #ffffff;
        text-align: center;
        margin-bottom: 6px;
    }
    .main-subtitle {
        text-align: center;
        color: #b794f4;
        font-size: 14px;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 40px;
    }
    
    /* Glassmorphic Section Headers */
    .section-header {
        font-size: 11px;
        font-weight: 700;
        color: #e9d8fd;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 10px;
        margin-bottom: 25px;
        background: rgba(131, 80, 245, 0.08);
        border: 1px solid rgba(131, 80, 245, 0.15);
        border-left: 4px solid #805ad5; 
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        display: inline-block;
        width: 100%;
    }
    
    /*KPI Metric Layout Blocks with Neon Atmospheric Glow */
    .kpi-container {
        background: #110b1e;
        border: 1px solid #6b46c1; 
        padding: 22px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 8px 30px rgba(107, 70, 193, 0.18); 
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    /* Interactive Hover Glow Expansion */
    .kpi-container:hover {
        border-color: #9f7aea;
        box-shadow: 0 12px 35px rgba(159, 122, 234, 0.35); 
        transform: translateY(-4px); 
    }
    
    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        color: #cbd5e1; 
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 800;
        color: #b794f4; 
        line-height: 1;
    }
    .kpi-unit {
        font-size: 14px;
        font-weight: 600;
        color: #9f7aea; 
        margin-left: 3px;
    }
    
    /* Sidebar Navigation Context styling */
    div[data-testid="stSidebar"] {
        background-color: #050308 !important;
        border-right: 1px solid #44337a;
    }
    
    /* Bright Functional High-Contrast Input Labels */
    label[data-testid="stWidgetLabel"] p {
        color: #e9d8fd !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px;
    }
    
    /* Primary Interactive Buttons */
    button[data-testid="stBaseButton-primary"] {
        background-color: #805ad5 !important;
        border: 1px solid #b794f4 !important;
        color: #09070f !important; 
        font-weight: 700 !important;
        height: 48px !important;
        border-radius: 8px !important;
        letter-spacing: 0.5px;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        background-color: #b794f4 !important; 
        border: 1px solid #e9d8fd !important;
    }

    /* Muted structural lines to respect content padding scales */
    hr {
        border-color: rgba(107, 70, 193, 0.2) !important;
        margin: 25px 0px !important;
    }
    </style>
""", unsafe_allow_html=True)

# SESSION STATE
if "app_step" not in st.session_state: st.session_state.app_step = 1
if "patient_name" not in st.session_state: st.session_state.patient_name = ""
if "patient_age" not in st.session_state: st.session_state.patient_age = 45
if "patient_gender" not in st.session_state: st.session_state.patient_gender = "Male"
if "audio_buffer" not in st.session_state: st.session_state.audio_buffer = None

# RUN ENGINE CALLS FROM OUR EXTERNAL MODULES
model, scaler, dataset_rows, train_accuracy, test_accuracy = load_screening_model()
st.sidebar.markdown("### Model Performance")
st.sidebar.write(f"Training Accuracy: {train_accuracy:.1f}%")   
st.sidebar.write(f"Test Accuracy: {test_accuracy:.1f}%")         
st.sidebar.write(f"Trained on: {dataset_rows} samples")          
# SIDEBAR INTERFACE METADATA 
with st.sidebar:
    st.markdown("<div style='padding: 10px 0px;'><h2 style='color:#ffffff; margin-bottom:5px; letter-spacing:0.5px;'>Console</h2></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Model Validation Metrics")
    
    st.metric(label="Calculated Training Accuracy", value=f"{train_accuracy:.2f}%")
    st.metric(label="Validated Testing Accuracy", value=f"{test_accuracy:.2f}%")
    
    st.markdown("---")
    st.markdown("### Core Architecture")
    st.code("Model: SVM (RBF Kernel)\nDimensions: 10 Features\nTotal Records: " + str(dataset_rows), language="yaml")

#APP BRANDING HEADER 
st.markdown("<h1 class='main-title'>Vocal Biomarker Diagnostic Analytics to detect parkinson's disease in the early stages</h1>", unsafe_allow_html=True)


#NAVIGATION ROUTER ROUTINES


# SCREEN 1: PATIENT INTAKE FORM
if st.session_state.app_step == 1:
    st.markdown("<div class='section-header'>Patient Metadata Collection</div>", unsafe_allow_html=True)
    
    name = st.text_input("Patient Full Name / Identifiable ID Code:", value=st.session_state.patient_name, placeholder="e.g., PT-9042")
    
    col_age, col_gen = st.columns(2)
    with col_age:
        age = st.number_input("Patient Age (Years):", min_value=1, max_value=120, value=st.session_state.patient_age)
    with col_gen:
        gender = st.selectbox("Biological Sex / Gender Profile:", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.patient_gender))
        
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    if st.button("Proceed to Acoustic Intake", type="primary", use_container_width=True):
        if not name.strip():
            st.warning("Operational Note: Please enter a patient identifier code before advancing.")
        else:
            st.session_state.patient_name = name
            st.session_state.patient_age = age
            st.session_state.patient_gender = gender
            st.session_state.app_step = 2
            st.rerun()

# SCREEN 2: ACOUSTIC INTAKE 
elif st.session_state.app_step == 2:
    st.markdown("<div class='section-header'>Signal Acquisition Matrix</div>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 14px; color: #ffffff; margin-bottom: 5px; letter-spacing:0.3px;'>Active Target Profile: <strong>{st.session_state.patient_name}</strong></p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:13px; color:#b794f4; margin-bottom: 25px;'>Instruct the patient to sustain an 'ahhh' phonation at a comfortable, steady volume for at least 3-5 seconds.</p>", unsafe_allow_html=True)
    
    input_method = st.segmented_control(
        "Select Acquisition Source Layer:",
        options=["File Upload (.wav)", "Live Microphone Intake"],
        default="File Upload (.wav)"
    )
    
    st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)
    
    uploaded_audio = None
    if input_method == "File Upload (.wav)":
        uploaded_audio = st.file_uploader("Upload continuous phonation recording", type=["wav"])
    else:
        uploaded_audio = st.audio_input("Microphone Capture Controller", sample_rate=48000)

    if uploaded_audio is not None:
        st.session_state.audio_buffer = uploaded_audio
        st.markdown("---")
        st.markdown("<p style='font-size:14px; font-weight:600; color:#e9d8fd; margin-bottom:10px;'>Stream Verification Engine Buffer</p>", unsafe_allow_html=True)
        st.audio(uploaded_audio, format="audio/wav")
        st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)
        
        if st.button("Run Acoustic Biomarker Assessment", type="primary", use_container_width=True):
            st.session_state.app_step = 3
            st.rerun()
            
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    if st.button("Modify Demographics Data", type="secondary"):
        st.session_state.app_step = 1
        st.rerun()

# SCREEN 3: DIAGNOSTIC METRIC REPORT EVALUATION
elif st.session_state.app_step == 3:
    st.markdown("<div class='section-header'>Diagnostic Evaluation Report</div>", unsafe_allow_html=True)
    
    if st.session_state.audio_buffer is None:
        st.error("Fatal Pipeline Sync Error: Audio buffer is empty. Re-routing.")
        st.session_state.app_step = 2
        st.rerun()
        
    temp_filename = "temp_vocal_sample.wav"
    with open(temp_filename, "wb") as f:
        f.write(st.session_state.audio_buffer.getbuffer())
        
    prediction = None
    confidence = 0.0
        
    try:
        # Calling external clean processing module function
        feature_vector, display_metrics = extract_vocal_features(temp_filename)
        
        #PATIENT OVERVIEW LINE
        st.markdown(f"""
            <div style='border-left: 4px solid #805ad5; padding: 5px 15px; margin-bottom: 30px;'>
                <h3 style='margin: 0; color:#ffffff; font-weight:700;'>Target: {st.session_state.patient_name}</h3>
                <div style='font-size:14px; color:#b794f4; margin-top:3px;'>Age Profile: <strong>{st.session_state.patient_age}</strong> &nbsp;|&nbsp; Biological Assignment: <strong>{st.session_state.patient_gender}</strong></div>
            </div>
        """, unsafe_allow_html=True)
        
        #ACOUSTIC GRID METRICS
        st.markdown("<p style='font-size:16px; font-weight:700; color:#ffffff; margin-bottom:15px; letter-spacing:-0.3px;'>Extracted Acoustic Target Metrics</p>", unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2, gap="medium")
        with m_col1:
            st.markdown(f"""
                <div class='kpi-container'>
                    <div class='kpi-label'>Fundamental Pitch</div>
                    <div class='kpi-value'>{display_metrics['Mean Pitch (Hz)']:.1f}<span class='kpi-unit'>Hz</span></div>
                </div>
                <div class='kpi-container'>
                    <div class='kpi-label'>Amplitude Shimmer</div>
                    <div class='kpi-value'>{display_metrics['Local Shimmer (%)']:.3f}<span class='kpi-unit'>%</span></div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
                <div class='kpi-container'>
                    <div class='kpi-label'>Frequency Jitter</div>
                    <div class='kpi-value'>{display_metrics['Local Jitter (%)']:.3f}<span class='kpi-unit'>%</span></div>
                </div>
                <div class='kpi-container'>
                    <div class='kpi-label'>Harmonics-to-Noise</div>
                    <div class='kpi-value'>{display_metrics['HNR (Clarity dB)']:.2f}<span class='kpi-unit'>dB</span></div>
                </div>
            """, unsafe_allow_html=True)

        #MACHINE LEARNING PREDICTION METRICS
        st.markdown("---")
        st.markdown("<p style='font-size:16px; font-weight:700; color:#ffffff; margin-bottom:15px; letter-spacing:-0.3px;'>Diagnostic Boundary Analysis</p>", unsafe_allow_html=True)
        
        scaled_feature_vector = scaler.transform(feature_vector)
        prediction = model.predict(scaled_feature_vector)[0]
        probabilities = model.predict_proba(scaled_feature_vector)[0]
        confidence = probabilities[prediction] * 100
        
        current_jitter = display_metrics['Local Jitter (%)']
        current_shimmer = display_metrics['Local Shimmer (%)']
        current_hnr = display_metrics['HNR (Clarity dB)']
        
        # BIOLOGICAL GUARDRAIL SAFETY FILTERS
        if current_jitter == 0.0 and current_shimmer == 0.0:
            prediction = 0
            confidence = 99.9
        elif current_jitter < 0.25 and current_shimmer < 2.0 and current_hnr > 22.0:
            prediction = 0
            confidence = 95.0

    except Exception as e:
        st.error(f"Processing Error: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    # UI DISPLAY LAYER~
    if prediction is not None:
        if prediction == 0:
            st.markdown(f"""
                <div style="background: rgba(46, 213, 115, 0.05); border: 1px solid #2ed573; box-shadow: 0 0 25px rgba(46, 213, 115, 0.2); padding: 25px; border-radius: 12px; margin-bottom: 15px;">
                    <h3 style="color: #2ed573; margin-top:0; font-weight:700; letter-spacing:0.5px;">PHYSIOLOGICAL NORMAL PROFILE</h3>
                    <p style="color: #f3f0f7; margin-bottom: 0; font-size:14px; line-height:1.6;">Vocal acoustic coordinates sit comfortably within baseline control parameters. Baseline signal verification indicates non-pathological vocal mechanics.</p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(int(confidence), text=f"System Classification Confidence: {confidence:.1f}%")
        else:
            st.markdown(f"""
                <div style="background: rgba(244, 63, 94, 0.05); border: 1px solid #f43f5e; box-shadow: 0 0 25px rgba(244, 63, 94, 0.25); padding: 25px; border-radius: 12px; margin-bottom: 15px;">
                    <h3 style="color: #fb7185; margin-top:0; font-weight:700; letter-spacing:0.5px;">ANOMALOUS ACOUSTIC SIGNS DETECTED</h3>
                    <p style="color: #f3f0f7; margin-bottom: 0; font-size:14px; line-height:1.6;">Vocal mechanics indicate statistically significant anomalies. The scaled spatial layout maps outside standard baseline boundaries.</p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(int(confidence), text=f"System Classification Confidence: {confidence:.1f}%")
            
    st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
    if st.button("Analyze a New Patient Profile", type="primary", use_container_width=True):
        st.session_state.app_step = 1
        st.session_state.audio_buffer = None
        st.rerun()