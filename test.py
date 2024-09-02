# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import tensorflow as tf
# from tensorflow.keras.layers import DepthwiseConv2D

# # Custom DepthwiseConv2D to ignore the 'groups' argument
# class CustomDepthwiseConv2D(DepthwiseConv2D):
#     def __init__(self, *args, **kwargs):
#         kwargs.pop('groups', None)
#         super().__init__(*args, **kwargs)

# # Register the custom layer
# tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# # Original code
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# offset = 20
# imgSize = 300
# folder = "Data/C"
# counter = 0
# labels = ["A", "B", "C"]

# while True:
#     success, img = cap.read()
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#         imgCropShape = imgCrop.shape
#         aspectRatio = h / w
#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap:wCal + wGap] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             print(prediction, index)
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#         cv2.rectangle(imgOutput, (x - offset, y - offset-50),
#                       (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
#         cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#         cv2.rectangle(imgOutput, (x-offset, y-offset),
#                       (x + w+offset, y + h+offset), (255, 0, 255), 4)
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#     cv2.imshow("Image", imgOutput)
#     cv2.waitKey(1)



import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D to ignore the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Register the custom layer
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# Original code
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0
labels = ["A", "B", "C"]  # Add more labels as needed

# Variables for word detection
current_word = ""
last_pred = None
last_time = time.time()
delay_between_letters = 1  # Time in seconds required to confirm a letter
delay_for_space = 1  # Time in seconds to add a space when no hand is detected

# Box size for displaying text
text_box_height = 100
text_box_bg_color = (0, 0, 0)  # Black background
text_color = (255, 255, 255)  # White text

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    img_width = imgOutput.shape[1]
    text_box = np.ones((text_box_height, img_width, 3), np.uint8) * np.array(text_box_bg_color, np.uint8)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Check if the prediction has been consistent for the delay period
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

        # Display the cropped images for debugging
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    else:
        # If no hands are detected, consider it as a space after the delay
        if time.time() - last_time > delay_for_space:
            current_word += " "
            last_pred = None
            last_time = time.time()

    # Draw the current word on the text box
    cv2.putText(text_box, current_word, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 3)

    # Reset the current_word if it exceeds the width of the text box
    text_size = cv2.getTextSize(current_word, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
    if text_size[0] > img_width - 40:  # Account for padding
        current_word = ""

    # Combine the original image and text box into a single display
    combined_output = np.vstack((imgOutput, text_box))

    # Display the combined output
    cv2.imshow("Output", combined_output)
    cv2.waitKey(1)
