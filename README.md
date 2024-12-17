# **Facial Emotion Recognition**
This project leverages deep learning to detect human emotions from facial expressions. It uses **Convolutional Neural Networks (CNN)** trained on the **FER2013** dataset to classify emotions into categories like Happy, Sad, Angry, and Surprise.

## **Features**
- Real-time emotion detection using a webcam.
- Emotion classification based on facial images.
- Use of **Haar Cascade** for face detection.
- Pre-trained **Keras model** for emotion prediction.

## **Installation**

### 1. **Clone the repository**
```bash
git clone https://github.com/SaiGawand12/Facial_Emotion_Recognition.git
cd Facial_Emotion_Recognition
```

### 2. **Set up a virtual environment**
```bash
cd src
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate     # For Windows
```

### 3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Run the application**
```bash
python src/face_detection.py
```
This will start the webcam and show the detected emotions in real-time.


### **Project Structure**
```
Facial_Emotion_Recognition/
├── src/
│   ├── venv/                            # Virtual environment for the project dependencies
│   ├── emotion_recognition.py           # Main script for emotion recognition model
│   ├── face_detection.py                # Script for face detection using Haar cascades
│   └── utils.py                         # Utility functions (e.g., preprocessing, helper functions)
├── model/
│   ├── emotion_model.h5                 # Pre-trained emotion recognition model (Keras model)
│   ├── haarcascade_frontalface_default.xml # Pre-trained face detection classifier
├── data/
│   ├── fer2013.csv                      # Dataset containing images and labels (FER2013 dataset)
├── requirements.txt                     # Python dependencies (e.g., TensorFlow, OpenCV, NumPy)
└── README.md                            # Project documentation (this file)
```

## **Usage**
- The webcam feed is used for real-time emotion recognition.
- The model classifies emotions like **Happy**, **Sad**, **Angry**, **Surprise**, etc.
- You can also pass custom images to the `emotion_recognition.py` for classification.

## **Technologies**
- **Python 3**
- **TensorFlow/Keras** for deep learning
- **OpenCV** for computer vision and webcam feed
- **NumPy** for data manipulation

## **Dataset**
The **FER2013** dataset, which contains facial images labeled with emotions, is used for training the model. It can be downloaded from various public repositories like Kaggle.

## **Contributing**
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes.
4. Push to the branch.
5. Create a Pull Request.

### Performance Metrics
The model achieves competitive accuracy for emotion recognition tasks on standard datasets.

## **License**
This project is licensed under the MIT License.

---
