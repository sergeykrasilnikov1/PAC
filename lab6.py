import random

import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
i = 0
motion_detected = False
red = False
cap = cv2.VideoCapture(0)
time = 0
timer = 50
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (not i):
        prev_frame = gray
        i+=1
    frame_delta = cv2.absdiff(prev_frame, gray)
    if frame_delta.sum() > 700000:
        motion_detected = True
    prev_frame = gray


    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]


    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)


    motion_map = cv2.merge((np.zeros_like(thresh), thresh, np.zeros_like(thresh)))

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    frame_without_contours = frame.copy()
    if motion_detected:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if time==timer or red:
        cv2.putText(frame, "Red color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        red = True
        time-=1
    else:
        frame = frame_without_contours
        cv2.putText(frame, "Green color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Face Detection', frame)
    cv2.imshow('Motion Map', motion_map)
    motion_detected = False
    if not red:
        time+=1
    if time==0:
        red = False

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
