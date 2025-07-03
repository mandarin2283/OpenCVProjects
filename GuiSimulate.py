import pyautogui as gui

import cv2

import HandTrackingClass as htm


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = htm.HandDetector()

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)
    lm_list = detector.find_pos(img)

    if len(lm_list)!=0:
        length = detector.find_distance(lm_list[4][1:],lm_list[8][1:])

        if length<50:
            gui.press('space')

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break