# import cv2
# import tkinter as tk
# from tkinter import Label, Button, StringVar
# from PIL import Image, ImageTk
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import time
# import tensorflow as tf
# from tensorflow.keras.layers import DepthwiseConv2D
# import pyttsx3  # Import the pyttsx3 library

# # Custom DepthwiseConv2D to ignore the 'groups' argument
# class CustomDepthwiseConv2D(DepthwiseConv2D):
#     def __init__(self, *args, **kwargs):
#         kwargs.pop('groups', None)
#         super().__init__(*args, **kwargs)

# # Register the custom layer
# tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# # Initialize model and hand detector
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("keras_model.h5", "labels.txt")
# offset = 20
# imgSize = 300
# labels = ["A", "B", "C", "H", "I", "K", "L", "S", "T"]  # Labels from your model

# # Variables for word detection
# current_word = ""
# last_pred = None
# last_time = time.time()
# delay_between_letters = 1  # Time in seconds required to confirm a letter
# delay_for_space = 1  # Time in seconds to add a space when no hand is detected

# # Initialize voice engine
# engine = pyttsx3.init()

# # Initialize Tkinter window
# root = tk.Tk()
# root.title("Sign Language Detection")
# root.geometry("1000x700")
# root.configure(bg="#2E2E2E")  # Dark background for the window

# # Label to display recognized text
# recognized_text = StringVar()
# text_label = Label(root, textvariable=recognized_text, font=("Helvetica", 36, "bold"), bg="#2E2E2E", fg="#FFFFFF")
# text_label.pack(pady=20, fill="both")

# # Canvas to display video feed
# video_frame = Label(root, bg="#2E2E2E")
# video_frame.pack(pady=10)

# # List to store hand positions for movement tracking
# hand_positions = []

# # Function to update video feed and perform detection
# def update_frame():
#     global last_pred, last_time, current_word, hand_positions

#     success, img = cap.read()
#     if success:
#         imgOutput = img.copy()
#         hands, img = detector.findHands(img)
#         img_width = imgOutput.shape[1]

#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand['bbox']

#             # Track hand's center position
#             center_x = x + w // 2
#             center_y = y + h // 2

#             # Store the position of the hand's center
#             hand_positions.append((center_x, center_y))

#             # Keep the list size small to only track the last few frames
#             if len(hand_positions) > 20:  # Store the last 20 frames
#                 hand_positions.pop(0)

#             # Draw the trajectory of the hand
#             for i in range(1, len(hand_positions)):
#                 # Line thickness depends on speed of movement (difference between points)
#                 thickness = int(math.sqrt(i) * 2)
#                 cv2.line(imgOutput, hand_positions[i - 1], hand_positions[i], (0, 255, 0), thickness)

#             # Your existing logic for hand detection and prediction
#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#             aspectRatio = h / w

#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap:wCal + wGap] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap:hCal + hGap, :] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)

#             # Your existing logic for word detection
#             if last_pred == labels[index]:
#                 if time.time() - last_time > delay_between_letters:
#                     current_word += labels[index]
#                     last_pred = None  # Reset last_pred to start detecting the next letter
#             else:
#                 last_pred = labels[index]
#                 last_time = time.time()

#             # Draw the predicted letter
#             cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
#                           (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
#             cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#             cv2.rectangle(imgOutput, (x - offset, y - offset),
#                           (x + w + offset, y + h + offset), (255, 0, 255), 4)

#         else:
#             # If no hands are detected, add a space after the delay
#             if time.time() - last_time > delay_for_space:
#                 current_word += " "
#                 last_pred = None
#                 last_time = time.time()

#         # Update the recognized text
#         recognized_text.set(current_word)

#         # Convert image to RGB and display it in the Tkinter window
#         imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(imgOutput)
#         imgtk = ImageTk.PhotoImage(image=img)
#         video_frame.imgtk = imgtk
#         video_frame.configure(image=imgtk)

#     root.after(10, update_frame)

# # Function to speak text
# def speak(text):
#     engine.say(text)
#     engine.runAndWait()

# # Start button to begin detection
# def start_detection():
#     speak("Program has started")
#     update_frame()

# # Stop button to end detection
# def stop_detection():
#     speak("Program has stopped")
#     cap.release()
#     root.quit()

# # Reset button to clear the recognized text
# def reset_detection():
#     global current_word
#     current_word = ""
#     recognized_text.set("")
#     speak("Text has been reset")

# # Add buttons for start, stop, and reset
# button_frame = tk.Frame(root, bg="#2E2E2E")
# button_frame.pack(pady=20)

# start_button = Button(button_frame, text="Start", font=("Helvetica", 18, "bold"), bg="#4CAF50", fg="#FFFFFF", command=start_detection)
# start_button.grid(row=0, column=0, padx=10, pady=5)

# reset_button = Button(button_frame, text="Reset", font=("Helvetica", 18, "bold"), bg="#FFC107", fg="#FFFFFF", command=reset_detection)
# reset_button.grid(row=0, column=1, padx=10, pady=5)

# stop_button = Button(button_frame, text="Stop", font=("Helvetica", 18, "bold"), bg="#F44336", fg="#FFFFFF", command=stop_detection)
# stop_button.grid(row=0, column=2, padx=10, pady=5)

# # Run the Tkinter event loop
# root.mainloop()

# import cv2
# import streamlit as st
# from PIL import Image
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import time
# import tensorflow as tf
# from tensorflow.keras.layers import DepthwiseConv2D

# # Custom DepthwiseConv2D to ignore the 'groups' argument
# class CustomDepthwiseConv2D(DepthwiseConv2D):
#     def __init__(self, *args, **kwargs):
#         kwargs.pop('groups', None)
#         super().__init__(*args, **kwargs)

# # Register the custom layer
# tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# # Initialize model and hand detector
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("keras_model.h5", "labels.txt")
# offset = 20
# imgSize = 300
# labels = ["A", "B", "C","D", "E","F", "G", "H", "I","J", "K", "L","M","N","O","P","Q","R", "S", "T","U","V","W","X","Y","Z"]  # Labels from your model

# # Variables for word detection
# current_word = ""
# last_pred = None
# last_time = time.time()
# delay_between_letters = 1  # Time in seconds required to confirm a letter
# delay_for_space = 1  # Time in seconds to add a space when no hand is detected

# # Streamlit Layout
# st.title("Sign Language Detection")

# # Placeholders for buttons and video feed
# start_button = st.button("Start Detection")
# reset_button = st.button("Reset Detection")
# stop_button = st.button("Stop Detection")
# video_placeholder = st.empty()
# text_placeholder = st.empty()

# # Function to capture and process video frames
# def capture_and_detect():
#     global last_pred, last_time, current_word

#     while cap.isOpened():
#         success, img = cap.read()
#         if not success:
#             break
        
#         imgOutput = img.copy()
#         hands, img = detector.findHands(img)

#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand['bbox']

#             # Prepare the image for prediction
#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#             imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#             aspectRatio = h / w

#             if aspectRatio > 1:
#                 k = imgSize / h
#                 wCal = math.ceil(k * w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 wGap = math.ceil((imgSize - wCal) / 2)
#                 imgWhite[:, wGap:wCal + wGap] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap:hCal + hGap, :] = imgResize
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)

#             # Handle word detection
#             if last_pred == labels[index]:
#                 if time.time() - last_time > delay_between_letters:
#                     current_word += labels[index]
#                     last_pred = None  # Reset last_pred to start detecting the next letter
#             else:
#                 last_pred = labels[index]
#                 last_time = time.time()

#             # Draw the predicted letter
#             cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
#                           (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
#             cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#             cv2.rectangle(imgOutput, (x - offset, y - offset),
#                           (x + w + offset, y + h + offset), (255, 0, 255), 4)
#         else:
#             if time.time() - last_time > delay_for_space:
#                 current_word += " "
#                 last_pred = None
#                 last_time = time.time()

#         # Update recognized text
#         text_placeholder.text(f"Recognized Text: {current_word}")

#         # Display the video frame in Streamlit
#         imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
#         frame_image = Image.fromarray(imgRGB)
#         video_placeholder.image(frame_image)

# # Reset detection
# if reset_button:
#     current_word = ""
#     text_placeholder.text("Recognized Text: ")

# # Start detection
# if start_button:
#     capture_and_detect()

# # Stop detection
# if stop_button:
#     cap.release()
#     st.stop()

import cv2
import streamlit as st
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from deepface import DeepFace
import mediapipe as mp

# Custom DepthwiseConv2D to ignore the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Register the custom layer
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# Initialize models
cap = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]  # Labels from your model

# Variables for word detection
current_word = ""
last_pred = None
last_time = time.time()
delay_between_letters = 1  # Time in seconds required to confirm a letter
delay_for_space = 1  # Time in seconds to add a space when no hand is detected

# Streamlit Layout
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #4CAF50;
    }
    .header {
        text-align: center;
        font-size: 24px;
        color: #555555;
        margin-bottom: 20px;
    }
    .button {
        display: block;
        width: 200px;
        margin: 0 auto;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px;
        cursor: pointer;
    }
    .button:hover {
        background-color: #45a049;
    }
    .video-container {
        text-align: center;
        margin-top: 20px;
    }
    .text-container {
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #FF5722;
        background-color: #F0F0F0;
        padding: 10px;
        border-radius: 10px;
    }
    .expression-container {
        text-align: center;
        font-size: 24px;
        color: #3F51B5;
        margin-top: 10px;
        background-color: #F0F0F0;
        padding: 10px;
        border-radius: 10px;
    }
    .movement-container {
        text-align: center;
        font-size: 24px;
        color: #009688;
        margin-top: 10px;
        background-color: #F0F0F0;
        padding: 10px;
        border-radius: 10px;
    }
    .sidebar {
        background-color: #f7f7f7;
        padding: 10px;
    }
    .main {
        background-color: #e0f7fa;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='title'>Cognitive Creators</div>", unsafe_allow_html=True)
st.markdown("<div class='header'>Choose Detection Type 🤟😊</div>", unsafe_allow_html=True)

# Sidebar for selecting detection type
with st.sidebar:
    st.markdown("<div class='sidebar'>", unsafe_allow_html=True)
    option = st.selectbox(
        "Select Detection Type",
        ["Sign Language Detection", "Facial Expression Recognition", "Body Movement Detection"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Placeholders for buttons and video feed
col1, col2, col3 = st.columns(3)
with col1:
    start_button = st.button("Start Detection", key='start', help="Click to start detection", use_container_width=True)
with col2:
    reset_button = st.button("Reset Detection", key='reset', help="Click to reset detection", use_container_width=True)
with col3:
    stop_button = st.button("Stop Detection", key='stop', help="Click to stop detection", use_container_width=True)

video_placeholder = st.empty()
text_placeholder = st.empty()
expression_placeholder = st.empty()
movement_placeholder = st.empty()

def capture_and_detect():
    global last_pred, last_time, current_word

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break
        
        imgOutput = img.copy()

        if option == "Sign Language Detection":
            hands, img = hand_detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                # Prepare the image for prediction
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                # Handle word detection
                if last_pred == labels[index]:
                    if time.time() - last_time > delay_between_letters:
                        current_word += labels[index]
                        last_pred = None  # Reset last_pred to start detecting the next letter
                else:
                    last_pred = labels[index]
                    last_time = time.time()

                # Draw the predicted letter
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)
            else:
                if time.time() - last_time > delay_for_space:
                    current_word += " "
                    last_pred = None
                    last_time = time.time()

            # Update recognized text with larger font
            text_placeholder.markdown(f"<div class='text-container'>{current_word}</div>", unsafe_allow_html=True)
        
        elif option == "Facial Expression Recognition":
            # Facial expression detection
            try:
                result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                expressions = result[0]['emotion']
                dominant_emotion = max(expressions, key=expressions.get)
                expression_placeholder.markdown(f"<div class='expression-container'>Facial Expression: {dominant_emotion} 😄</div>", unsafe_allow_html=True)
            except Exception as e:
                expression_placeholder.markdown(f"<div class='expression-container'>Facial Expression: Error 😕</div>", unsafe_allow_html=True)
            
        elif option == "Body Movement Detection":
            # Pose detection
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            
            if results.pose_landmarks:
                # Draw pose landmarks on the image
                landmarks = results.pose_landmarks.landmark
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                
                # Check for specific body movements
                movement_message = "No significant movement detected."
                if left_wrist.y < left_elbow.y and right_wrist.y < right_elbow.y:
                    movement_message = "Both hands are raised."
                elif left_wrist.y < left_elbow.y:
                    movement_message = "Left hand is raised."
                elif right_wrist.y < right_elbow.y:
                    movement_message = "Right hand is raised."
                elif abs(left_wrist.x - right_wrist.x) < 0.1 and abs(left_wrist.y - right_wrist.y) < 0.1:
                    movement_message = "Hands are joined (Namaste)."
                    
                movement_placeholder.markdown(f"<div class='movement-container'>{movement_message}</div>", unsafe_allow_html=True)
        
        # Display the video frame in Streamlit
        imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(imgRGB)
        video_placeholder.image(frame_image, use_column_width=True)

# Reset detection
if reset_button:
    current_word = ""
    text_placeholder.markdown("<div class='text-container'>Recognized Text: </div>", unsafe_allow_html=True)
    expression_placeholder.markdown("<div class='expression-container'>Facial Expression: </div>", unsafe_allow_html=True)
    movement_placeholder.markdown("<div class='movement-container'>Body Movement: </div>", unsafe_allow_html=True)

# Start detection
if start_button:
    capture_and_detect()

# Stop detection
if stop_button:
    cap.release()
    st.stop()







