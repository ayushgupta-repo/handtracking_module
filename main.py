import cv2
import mediapipe as mp
import time

# Creating web cam to open using VideoCapture class of cv2 and for using laptop's webcam provide index 0

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    cv2. imshow('Image', img)
    cv2.waitKey(1)
