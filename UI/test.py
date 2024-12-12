import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import librosa
import pyaudio
import pandas as pd
import time
import soundfile as sf
import io
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""<style>
    .stApp { background-color: #ffffff; }
    .main { padding: 2rem; }
    .stButton>button { border-radius: 20px; padding: 0.5rem 2rem; background-color: #4CAF50; color: white; font-weight: bold; border: none; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #45a049; transform: translateY(-2px); }
    .emotion-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px 0; }
    .song-card { background-color: #00000; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); margin: 8px 0; transition: transform 0.2s ease; }
    .song-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); }
    .step-header { color: #1f77b4; margin-bottom: 1rem; }
    .progress-container { margin: 1rem 0; padding: 1rem; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
    .disabled-step { opacity: 0.6; pointer-events: none; }
    .success-message { color: #4CAF50; font-weight: bold; padding: 10px; border-radius: 5px; margin: 10px 0; }
    .error-message { color: #f44336; font-weight: bold; padding: 10px; border-radius: 5px; margin: 10px 0; }
</style>""", unsafe_allow_html=True)

# Load pre-trained models
@st.cache_resource
def load_models():
    face_model = tf.keras.models.load_model('C://Users//uday//OneDrive//Desktop//EMR//Fuse//testData//Models//face_cnn9.h5')
    audio_model = tf.keras.models.load_model('C://Users//uday//OneDrive//Desktop//EMR//Fuse//testData//Models//Emotion_Voice_Detection_Model1.h5')
    return face_model, audio_model

face_model, audio_model = load_models()

# Emotion labels with corresponding emojis
emotions = {
    'Neutral': '😐',
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠'
}

# Initialize session state
session_state_vars = {
    'camera_active': False,
    'captured_face': None,
    'audio_recording_active': False,
    'audio_input': None,
    'audio_sample_rate': None,
    'face_prediction': None,
    'audio_prediction': None,
    'capture_timestamp': None,
    'step': 1,
    'analyzing': False
}

for var, default_value in session_state_vars.items():
    if var not in st.session_state:
        st.session_state[var] = default_value

# Helper functions
def preprocess_frame(frame, target_size=(128, 128)):
    """Preprocess the captured frame for the face model."""
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

def preprocess_audio(audio, sr=22050, n_mfcc=40):
    """Extract MFCC features from audio for model input."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs_mean, axis=0)

def dynamic_weight_adjustment(face_probs, audio_probs):
    """Adjust weights dynamically based on model confidence."""
    face_confidence = np.max(face_probs)
    audio_confidence = np.max(audio_probs)
    total_confidence = face_confidence + audio_confidence
    face_weight = face_confidence / total_confidence
    audio_weight = audio_confidence / total_confidence
    return face_weight, audio_weight

def weighted_emotion_prediction(face_probs, audio_probs, threshold=0.5):
    """Combine face and audio predictions with dynamic weighting."""
    face_weight, audio_weight = dynamic_weight_adjustment(face_probs, audio_probs)
    final_probs = face_weight * face_probs + audio_weight * audio_probs
    if np.max(final_probs) < threshold:
        return "Neutral", final_probs
    else:
        final_emotion_index = np.argmax(final_probs)
        return list(emotions.keys())[final_emotion_index], final_probs

def capture_image(cap):
    """Capture and process a frame from the camera."""
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.session_state.captured_face = frame_rgb
        st.session_state.capture_timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.camera_active = False
        st.session_state.step = 2
        return True
    return False

def recommend_songs(emotion):
    """Get song recommendations based on detected emotion."""
    try:
        df = pd.read_csv("C://Users//uday//OneDrive//Desktop//EMR//UI//songs.csv")
        if 'MoodsStrSplit' not in df.columns:
            raise ValueError("The 'MoodsStrSplit' column is missing in the CSV file.")
        
        recommended_songs = df[df['MoodsStrSplit'].str.contains(emotion, case=False, na=False)]
        if emotion.lower() in ['angry', 'sad']:
            additional_songs = df[df['MoodsStrSplit'].str.contains("happy", case=False, na=False)]
            recommended_songs = pd.concat([recommended_songs, additional_songs])
        
        return recommended_songs[['Artist', 'Title', 'SampleURL']].head(10)
    except Exception as e:
        st.error(f"Error in song recommendations: {str(e)}")
        return pd.DataFrame()

# App title and description
st.title("🎭 Emotion Recognition and Music Recommendation System")
st.markdown("### Let's discover your emotion and find the perfect music for your mood!")

# Progress tracking
#progress_placeholder = st.empty()
#progress_value = (st.session_state.step - 1) * 33.33
#progress_placeholder.progress(int(progress_value))

# Create three columns for layout
col1, col2, col3 = st.columns([1, 1, 1])

# Step 1: Facial Capture
with col1:
    st.markdown("### 📸 Step 1: Facial Capture")
    camera_placeholder = st.empty()
    button_placeholder = st.empty()
    
    if st.session_state.step == 1:
        if button_placeholder.button("Toggle Camera", key="toggle_camera"):
            st.session_state.camera_active = not st.session_state.camera_active

        if st.session_state.camera_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Camera access failed!")
            else:
                capture_button = st.button("📸 Capture", key="capture_button")
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        
                        if capture_button:
                            if capture_image(cap):
                                st.success("✅ Image captured successfully!")
                                st.rerun()
                            break
                    else:
                        st.error("❌ Camera feed error!")
                        break
                cap.release()

    # Display captured image
    if st.session_state.captured_face is not None:
        camera_placeholder.image(st.session_state.captured_face, use_container_width=True)
        st.caption(f"Captured at {st.session_state.capture_timestamp}")

# Step 2: Audio Recording with PyAudio
with col2:
    st.markdown("### 🎤 Step 2: Voice Recording")
    if st.session_state.step >= 2:
        if st.button("🎙 Record Audio (5s)", key="record_audio"):
            with st.spinner("Recording..."):
                p = pyaudio.PyAudio()
                audio_data = []
                duration = 5
                sr = 22050
                stream = p.open(format=pyaudio.paInt16,
                                channels=1,
                                rate=sr,
                                input=True,
                                frames_per_buffer=1024)
                
                for _ in range(0, int(sr / 1024 * duration)):
                    data = stream.read(1024)
                    audio_data.append(data)

                stream.stop_stream()
                stream.close()
                p.terminate()
                
                audio_raw = b"".join(audio_data)
                audio_buffer = io.BytesIO(audio_raw)
                
                audio, sr = librosa.load(audio_buffer, sr=22050)
                st.session_state.audio_input = audio
                st.session_state.audio_sample_rate = sr
                
                st.success("✅ Audio recorded successfully!")
                st.session_state.step = 3
                st.experimental_rerun()

        if st.session_state.audio_input is not None:
            st.audio(st.session_state.audio_input, format='audio/wav')

# Step 3: Emotion Analysis and Music Recommendation
with col3:
    st.markdown("### 💡 Step 3: Emotion Analysis")
    if st.session_state.step >= 3:
        face_frame = preprocess_frame(st.session_state.captured_face) if st.session_state.captured_face else None
        audio_features = preprocess_audio(st.session_state.audio_input, sr=st.session_state.audio_sample_rate) if st.session_state.audio_input is not None else None

        if face_frame is not None and audio_features is not None:
            st.session_state.analyzing = True
            st.session_state.face_prediction = face_model.predict(face_frame)
            st.session_state.audio_prediction = audio_model.predict(audio_features)

            emotion, prob = weighted_emotion_prediction(st.session_state.face_prediction, st.session_state.audio_prediction)
            st.session_state.analyzing = False

            st.session_state.face_prediction = emotion
            st.session_state.audio_prediction = prob

            st.success(f"Detected Emotion: {emotion} {emotions[emotion]}")

            recommended_songs = recommend_songs(emotion)
            st.subheader("Recommended Songs:")
            if recommended_songs.empty:
                st.write("No songs found. Please try again.")
            else:
                for idx, row in recommended_songs.iterrows():
                    st.markdown(f"- **{row['Title']}** by {row['Artist']} - [Listen Here]({row['SampleURL']})")

# Add custom footer
st.markdown("---")
st.markdown("Built with ❤️ by Your Name - [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourprofile)")
