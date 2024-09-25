# initial set-up: install all required modules
import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image

# set the configuration of the application - title, disclaimer, and a horizontal line as a divider
st.set_page_config(page_title="Facial Detection")
st.title("Facial Detection System")
st.write("NOTE: THE FOLLOWING CLASSIFIER MAY RUN SLOW ON SOME DEVICES")
st.divider()

# run the following function at the start of the load-up to avoid longer loading times and store the result in the cache
@st.cache_resource
def detect_faces(frame):
    """ the following function takes in the parameter, frame, and uses the haarcascade frontalface classifier to detect (if any) faces in the given frame and draw a rectangle around them """
    # initializes a Haar Cascade classifier (object detection method )for a pre-trained model of faces and store the instance of the classifier in the variable face_cascade
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

    # convert the frame (rgb) to gray scaled since the classifier only works on gray scaled images
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # use the instance prev defined to detect any faces in the current frame and store them all in the variable faces as a list in the format of (x_coord, y_coord, with, height)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # pass in the gray scaled img, scale factor, and min-neighbors

    # loop through all the faces detected
    for (x, y, w, h) in faces:
        # use the inbuild rectangle function to draw a rectangle around detected face(s) on the original frame
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
    
    # return the original frame with the rectangle drawn on it
    return frame

# set the config for the upload file section
st.header("Upload File")
st.write("Click the button below to upload a file containing a face(s) and watch the magic happen!")

# set the frame in which the faces will be detected from as the image uploaded to the site
frame = st.file_uploader("Upload file", label_visibility="hidden")

# check if a file was uploaded
if frame is not None:
    # if it was, then the button "start detecting" will pop up
    img_detect = st.button("Start detecting")

    # check if "start detecting" is clicked
    if img_detect:
        # open the file that was uploaded
        frame = Image.open(frame)
        # pass this frame into the detect_faces function, but change the frame to a NumPy array, which is required in order to process the image
        frame = detect_faces(np.array(frame))
    
    # display the new frame/image with the detected faces
    st.image(frame)

st.divider()

# set the config for the open webcam section
st.header("Open Webcam")
st.write("Click the start recording button to detect faces in real time!")

# open a new video capture using the computers inbuilt camera
cap = cv.VideoCapture(0)
window = st.empty() # start with the current window (which will display the webcam footage) to be empty, but will be updated later on

# format the start and stop recording buttons to appear on opposite sides of the screen through the use of columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    # declare a start recording button
    start_button = st.button("Start Recording")
with col4:
    # declare a stop recording button
    stop_button = st.button("Stop Recording")

# check if the start recording button was clicked
if start_button:
    # run the following code as long as the stop recording button isn't clicked
    while not stop_button:
        # get/take a snapshot of the current frame in the webcam and store it in the variable frame
        _, frame = cap.read()

        # pass this frame into the detect_faces function and save the new frame with the detected faces into the frame variable
        frame = detect_faces(frame)
        
        # change the image format from BGR back to RGB in order to display it
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # update the window (which was pre empty) with the new frame
        window.image(frame, channels="RGB")

    # when the loop ends  (stop recording button was clicked), then close the camera and close the window
    cap.release()
    cv.destroyAllWindows()