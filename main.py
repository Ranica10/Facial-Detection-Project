import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Facial Detection")
st.title("Facial Detection System")
st.write("NOTE: THE FOLLOWING CLASSIFIER MAY RUN SLOW ON SOME DEVICES")
st.divider()

@st.cache_resource
def detect_faces(frame):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # turn from coloured frame to gray scaled

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # pass in gray scaled img, scale factor, and min-neighbors
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5) # draw rectangle around detected face
    
    return frame

st.header("Upload File")
st.write("Click the button below to upload a file containing a face(s) and watch the magic happen!")
file = st.file_uploader("Upload file", label_visibility="hidden")

if file is not None:
    img_detect = st.button("Start detecting")

    if img_detect:
        file = Image.open(file)
        file = detect_faces(np.array(file))
    
    st.image(file)

st.divider()

st.header("Open Webcam")
st.write("Click the start recording button to detect faces in real time!")

# open a new video capture using the computers inbuilt camera
cap = cv.VideoCapture(0)
window = st.empty()

col1, col2, col3, col4 = st.columns(4)
with col1:
    start_button = st.button("Start Recording")
with col4:
    stop_button = st.button("Stop Recording")

if start_button:
    while not stop_button:
        _, frame = cap.read()

        frame = detect_faces(frame)
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        window.image(frame, channels="RGB")

    cap.release()
    cv.destroyAllWindows()