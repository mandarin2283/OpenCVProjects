import pyautogui
import pyautogui as pyauto

import cv2
import numpy as np

import HandTrackingClass as htm

width_camera = 640
height_camera = 480
frame = 100
smooth = 10
time = 0
plocx,plocy = 0,0
clockx,clocky = 0,0

cap = cv2.VideoCapture(0)
cap.set(3,width_camera)
cap.set(4,height_camera)
detector = htm.HandDetector()

screen_w,screen_h = pyautogui.size()
MARGIN = 5

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)
    lm_list = detector.find_pos(img)

    if len(lm_list)!=0:
        length_to_space = detector.find_distance(lm_list[4][1:],lm_list[8][1:])
        x1,y1 = lm_list[8][1:]
        x2, y2 = lm_list[12][1:]
        bent_list = detector.is_bent(lm_list)

        if bent_list[1]==0:
            x3 = np.interp(x1,(frame,width_camera-frame),
                           (0,screen_w))
            y3 = np.interp(y1,(frame,height_camera-frame),
                           (0,screen_h))
            clockx = plocx + (x3 - plocx)/smooth
            clocky = plocy + (y3 - plocy)/smooth
            pyauto.moveTo(screen_w - clockx,clocky)
            plocx,plocy = clockx,clocky

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break