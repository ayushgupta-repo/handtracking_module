import cv2
import mediapipe as mp
import time

# creating class


class handleDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modeComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # creating object to create finger locations
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.modeComplex, self.detectionCon, self.trackCon)

        # to draw line between the landmarks predefined function is used
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        # Converting image from BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Below will process frames of the image
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:

                if draw:

                    # using mediapipe we can create those landmark points and joining those points using line
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

        # # getting id of landmark and x and y positions in form of pixels values

        # for id, lm in enumerate(handLms.landmark):
        #     h, w, c = img.shape

        #     cx, cy = int(lm.x*w), int(lm.y*h)

        #     print(id, cx, cy)

        #     # drawing circle on landmarks
        #     cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)


def main():

    pTime = 0
    cTime = 0

    # Creating web cam to open using VideoCapture class of cv2 and for using laptop's webcam provide index 0
    cap = cv2.VideoCapture(0)

    # creating detector object
    detector = handleDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

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


if __name__ == '__main__':
    main()
