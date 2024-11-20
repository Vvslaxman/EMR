import cv2
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd

# Load pre-trained models
face_model = tf.keras.models.load_model('C://Users//uday//OneDrive//Desktop//EMR//Fuse//testData//Models//face_cnn9.h5')
audio_model = tf.keras.models.load_model('C://Users//uday//OneDrive//Desktop//EMR//Fuse//testData//Models//Emotion_Voice_Detection_Model1.h5')

# Emotion labels
emotions = ['Neutral', 'Happy', 'Sad', 'Angry']

# Preprocess the image
def preprocess_frame(frame, target_size=(128, 128)):
    frame_resized = cv2.resize(frame, target_size)  # Resize to match the model's input
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    frame_expanded = np.expand_dims(frame_normalized, axis=0)  # Add batch dimension
    return frame_expanded

# Preprocess the audio
def preprocess_audio(duration=3, sr=22050, n_mfcc=40):
    print("Recording Audio...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is complete
    audio = audio.flatten()  # Flatten the audio array
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # Calculate mean of MFCCs
    return np.expand_dims(mfccs_mean, axis=0)  # Add batch dimension

# Dynamic weight adjustment based on confidence
def dynamic_weight_adjustment(face_probs, audio_probs):
    face_confidence = np.max(face_probs)
    audio_confidence = np.max(audio_probs)
    total_confidence = face_confidence + audio_confidence
    face_weight = face_confidence / total_confidence
    audio_weight = audio_confidence / total_confidence
    return face_weight, audio_weight

# Weighted emotion prediction with confidence thresholding
def weighted_emotion_prediction(face_probs, audio_probs, threshold=0.5):
    face_weight, audio_weight = dynamic_weight_adjustment(face_probs, audio_probs)
    final_probs = face_weight * face_probs + audio_weight * audio_probs
    if np.max(final_probs) < threshold:
        return "Neutral", final_probs  # Default to "Neutral" if confidence is too low
    else:
        final_emotion_index = np.argmax(final_probs)
        return emotions[final_emotion_index], final_probs

# Step 1: Capture facial data
cap = cv2.VideoCapture(0)  # Open webcam (0 is the default device)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

print("Press 'c' to capture a face image and proceed.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Show the live webcam feed
    cv2.imshow('Webcam Feed - Press "c" to Capture', frame)

    # Wait for user to press 'c' to capture the frame
    if cv2.waitKey(1) & 0xFF == ord('c'):
        print("Face image captured.")
        face_input = preprocess_frame(frame)
        break

cap.release()
cv2.destroyAllWindows()

# Step 2: Capture audio data
audio_input = preprocess_audio(duration=3)

# Step 3: Get predictions from both models
face_probs = face_model.predict(face_input)
audio_probs = audio_model.predict(audio_input)

# Step 4: Fuse predictions and get the final emotion
predicted_emotion, final_probs = weighted_emotion_prediction(face_probs, audio_probs)

# Step 5: Output the prediction
print("Face Model Predicted Probabilities:", face_probs)
print("Audio Model Predicted Probabilities:", audio_probs)
print("Predicted Emotion:", predicted_emotion)
