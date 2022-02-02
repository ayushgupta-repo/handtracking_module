import cv2
import mediapipe as mp
import time

# Creating web cam to open using VideoCapture class of cv2 and for using laptop's webcam provide index 0

cap = cv2.VideoCapture(0)

# creating object to create finger locations
mpHands = mp.solutions.hands
hands = mpHands.Hands()

# to draw line between the landmarks predefined function is used
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()

    # Converting image from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Below will process frames of the image
    results = hands.process(imgRGB)

    # Extracting hands if multiple present

    # Simple result will only return class but to do detection of hand we have to add results.multi_hand_landmarks
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:

        # handLms = hand numbers
        for handLms in results.multi_hand_landmarks:

            # using mediapipe we can create those landmark points and joining those points using line
            mpDraw.draw_landmarks(img, handLms)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
