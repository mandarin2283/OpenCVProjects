import math
import time
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL

import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume

import HandTrackingClass as htm

WEIDTH_CAMERA, HEIGHT_CAMERA = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3,WEIDTH_CAMERA)
cap.set(4,HEIGHT_CAMERA)
cur_time = 0
prev_time = 0

detector = htm.HandDetector()

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume = cast(interface,POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol_range = vol_range[0]
max_vol_range = vol_range[1]
vol_bar = 400

while True:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    lm_list = detector.find_pos(img,draw=False)

    if len(lm_list)!=0:
        x1,y1 = lm_list[4][1],lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        cv2.circle(img,(x1,y1),15,(0,0,255),thickness=-1)
        cv2.circle(img,(x2,y2),15,(0,0,255),thickness=-1)
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), thickness=-1)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)

        length = math.hypot(x2-x1,y2-y1)

        vol = np.interp(length,[50,200],[min_vol_range,max_vol_range])
        vol_bar = np.interp(length, [50, 200], [400,150])
        volume.SetMasterVolumeLevel(vol,None)
    cv2.rectangle(img,(50,150),(85,400),(255,0,0),3)
    cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), -1)


    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, f'FPS{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
