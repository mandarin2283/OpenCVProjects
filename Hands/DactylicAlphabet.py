import cv2
import numpy as np

from Hands import HandTrackingClass as htm

WIDTH_CAMERA, HEIGHT_CAMERA = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3,WIDTH_CAMERA)
cap.set(4,HEIGHT_CAMERA)

detector = htm.HandDetector()

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)
    lm_list = detector.find_pos(img)
    if len(lm_list)!=0:
        pass

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
