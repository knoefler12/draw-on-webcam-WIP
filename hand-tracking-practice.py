import cv2
import numpy as np
import mediapipe as mp
import time

# Reading video
# Give integer value to capture webcam, else give path
capture = cv2.VideoCapture(0)
# to read videos u need to use a while loop

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
previousTime = 0
currentTime = 0
while True:
    # capure.read returns a bool that show wether read or not, and a frame. since it reads frame by frame
    isTrue, frame = capture.read()
    # Frame needs to be converted to rgb because hands only uses rgb
    frameRGB = cv2.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    # print(results.multi_hand_landmark) # detects if hand is in frame

    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            # id is the index number we are getting from our hand landmarks
            for id, landmark in enumerate(handLandmark.landmark):
                #print(id,landmark)
                height, width, channel = frame.shape
                # gets position
                cx, cy = int(landmark.x*width), int(landmark.y*height)
                print(id, cx, cy)

                # draws circle on id 20
                if id == 20:
                    cv2.circle(frame, (cx, cy), 25,(255, 0, 255), -1)

            # Shows hand tracking on screen
            mpDraw.draw_landmarks(frame, handLandmark, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, 'press q to close webcam', (125, 450), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)

    cv.imshow('Frame', frame)

    if cv.waitKey(10) & 0xFF == ord('q') or cv.waitKey(10) & 0xFF == ord('Q'):
        break

capture.release()
cv.destroyAllWindows()
