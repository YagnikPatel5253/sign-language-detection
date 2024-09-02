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
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

# Custom DepthwiseConv2D to ignore the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Register the custom layer
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load the classifier
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
labels = ["A", "B", "C"]

# Variables to store the text
word = ""
sentence = ""

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

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
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        current_char = labels[index]
        
        # Append the current character to the word
        word += current_char

        # Display the current character, word, and sentence
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset + 300, y - offset-50 + 100), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, f'Character: {current_char}', (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(imgOutput, f'Word: {word}', (x, y + h + 40),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(imgOutput, f'Sentence: {sentence}', (x, y + h + 90),
                    cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)

        # Draw the bounding box around the hand
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    if key == ord(' '):  # Press space to add the current word to the sentence
        sentence += word + " "
        word = ""
    elif key == ord('c'):  # Press 'c' to clear the sentence
        sentence = ""

cap.release()
cv2.destroyAllWindows()




