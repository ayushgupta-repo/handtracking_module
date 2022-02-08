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

# setting time intervals
pTime = 0
cTime = 0

while True:
    success, img = cap.read()

    # Converting image from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Below will process frames of the image
    results = hands.process(imgRGB)

    # Extracting hands if multiple present

    # Simple result will only return class but to do detection of hand we have to add results.multi_hand_landmarks
    res = []

    # Extracting necessary informations
    # if results.multi_hand_landmarks:
    #     x = results.multi_hand_landmarks[0]
    #     y = results.multi_hand_landmarks[1]
    #     res.append(x)
    #     res.append(y)
    # print(results.multi_hand_landmarks)
    # print(res)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Accessing the object landmark by creating lm as in loop
            for lm in hand_landmarks.landmark:

                # fetching the height and width of an hand landmark image
                height, width, channel = img.shape

                # getting those dimensions converted into specific format
                cx, cy = int(lm.x * height), int(lm.y * width)

                # appending those values processed above
                res.append(cx)
                res.append(cy)
                # cv2.circle(img, (cx, cy), 10, (0, 0, 0), cv2.FILLED)

                print(res)
                res = []

    if results.multi_hand_landmarks:

        # handLms = hand numbers
        for handLms in results.multi_hand_landmarks:

            # using mediapipe we can create those landmark points and joining those points using line
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # creating frame rate

    # getting current time
    cTime = time.time()

    # creating fps
    fps = 1/(cTime-pTime)
    pTime = cTime

    # displaying fps

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
