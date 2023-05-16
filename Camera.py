from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow
from keras.preprocessing import image
import cv2
import numpy as np

import streamlit as st

# Load the pre-trained model and cascade classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classifier = load_model('Emotion_Detection.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to detect and classify emotions
def detect_emotions(frame):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)
        roi = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)
        roi = roi.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)

        prediction = classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Streamlit app
def main():
    st.title("Emotion Detector")

    # Open video capture
    cap = cv2.VideoCapture(0)

    # Continuously process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and classify emotions
        frame = detect_emotions(frame)

        # Display the processed frame
        st.image(frame, channels="BGR")

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
