import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array

# Load the pre-trained emotion recognition model
model = load_model('model/emotion_detection_model.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect faces
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Path to video file
video_path = 'C:/Users/SAI/OneDrive/Desktop/Desktop/Assignments/AGV/Projects/Facial_Emotion_Recognition/Videos/mixkit-girl-smiling-portrait-in-the-library-4756-hd-ready.mp4'
# video_path = 'C:/Users/SAI/OneDrive/Desktop/Desktop/Assignments/AGV/Projects/Facial_Emotion_Recognition/Videos/6026394_barbie_person_people_hb5e18d7dV0218079720p5000br.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Detect faces in the frame
    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (48, 48))

        # Normalize the pixel values to the range [0, 1]
        face_roi = face_roi.astype('float32') / 255
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Predict the emotion
        predictions = model.predict(face_roi)
        print(f"Predictions: {predictions}, Shape: {predictions.shape}")

        emotion_index = np.argmax(predictions)

        # Get the percentage of the predicted emotion
        confidence_percentage = predictions[0][emotion_index] * 100

        # Ensure the emotion index is within range of the emotion_labels list
        if emotion_index < len(emotion_labels):
            emotion = emotion_labels[emotion_index]
        else:
            emotion = "Unknown"

        # Create the label with emotion and confidence percentage
        label = f"{emotion}: {confidence_percentage:.2f}%"

        # Display the emotion label and confidence on the frame
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()