import cv2
import mediapipe as mp
import time

class HandTrackingDetector:
    # Sets default params to the same as in the mediapipe
    def __init__(self, mode=False, maxHands=2, modelComplex=1, detectionConfidence=0.5, trackingConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionConfidence, self.trackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=True):
        # Frame needs to be converted to rgb because hands only uses rgb
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLandmark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLandmark, self.mpHands.HAND_CONNECTIONS)

        return frame


    def findPostition(self, frame, handNumber=0, draw=True):
        landmarkList = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNumber]

            for id, landmark in enumerate(hand.landmark):
                # print(id,landmark)
                height, width, channel = frame.shape
                # gets position
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmarkList.append([id,cx,cy])
                # draws circle on id 20
                if draw:
                    if id == 20:
                        cv2.circle(frame, (cx, cy), 25, (255, 0, 255), -1)

        return landmarkList


def main():
    previousTime = 0
    currentTime = 0
    capture = cv2.VideoCapture(0)
    tracking = HandTrackingDetector()

    while True:
        # capure.read returns a bool that show wether read or not, and a frame. since it reads frame by frame
        isTrue, frame = capture.read()
        frame = tracking.findHands(frame)
        landmarkList = tracking.findPostition(frame,draw=False)
        if len(landmarkList) != 0:
            print(landmarkList[15])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'press q to close webcam', (125, 450), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(10) & 0xFF == ord('q') or cv2.waitKey(10) & 0xFF == ord('Q'):
            break


if __name__ == "__main__":
    main()
