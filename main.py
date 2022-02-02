import cv2
import mediapipe as mp
import time

# Creating web cam to open using VideoCapture class of cv2 and for using laptop's webcam provide index 0

cap = cv2.VideoCapture(0)

# creating object to create finger locations
mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:
    success, img = cap.read()

    # Converting image from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Below will process frames of the image
    results = hands.process(imgRGB)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
