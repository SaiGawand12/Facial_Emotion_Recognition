import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the pre-trained emotion recognition model
model = load_model('model\emotion_model.h5')

# Emotion Labels (Ensure this matches the number of classes your model was trained on)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Set Streamlit configuration
st.title("Real-Time Emotion Recognition")
st.write("This application detects emotions in real-time from your webcam feed.")

# Start the video stream
video_capture = cv2.VideoCapture(0)

# Function to detect faces
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Streamlit button to stop webcam
if st.button("Stop Webcam"):
    video_capture.release()
    cv2.destroyAllWindows()
    st.stop()

# Loop to capture frames from the webcam
while video_capture.isOpened():
    ret, frame = video_capture.read()
    
    # Detect faces in the frame
    faces = detect_faces(frame)
    
    for (x, y, w, h) in faces:
        # Preprocess face image
        face_region = frame[y:y + h, x:x + w]
        face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        face_region = cv2.resize(face_region, (48, 48))
        
        # Normalize the pixel values to the range [0, 1]
        face_region = face_region.astype('float32') / 255
        face_region = img_to_array(face_region)
        face_region = np.expand_dims(face_region, axis=0)

        # Predict emotion
        prediction = model.predict(face_region)
        
        # Get the index of the class with the highest probability
        emotion_index = np.argmax(prediction)
        
        # Get the percentage of the predicted emotion
        confidence_percentage = prediction[0][emotion_index] * 100
        
        # Ensure the emotion index is within range of the emotion_labels list
        if emotion_index < len(emotion_labels):
            emotion = emotion_labels[emotion_index]
        else:
            emotion = "Neutral"
        
        # Create the label with emotion and confidence percentage
        label = f"{emotion}: {confidence_percentage:.2f}%"
        
        # Display the emotion label and confidence on the frame
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert the frame to RGB format for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the frame in Streamlit
    st.image(frame_rgb, channels="RGB", use_column_width=True)

    # Break out of the loop if the Stop button is pressed
    if not video_capture.isOpened():
        break
