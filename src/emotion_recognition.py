import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load the pre-trained emotion recognition model
model = load_model('model/emotion_model.h5')

# Emotion Labels (Ensure this matches the number of classes your model was trained on)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect faces
def detect_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Start the video stream
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit the application.")

# Loop to capture frames from the webcam
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
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

    # Display the frame in a window
    cv2.imshow("Real-Time Emotion Recognition", frame)

    # Press 'q' to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
