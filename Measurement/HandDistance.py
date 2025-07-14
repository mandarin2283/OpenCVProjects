import cv2
import numpy as np

from Hands.HandTrackingClass import HandDetector


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = HandDetector()

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img,draw=False)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break