import time

import cv2
import mediapipe as mp
import numpy as np

import HandTrackingClass as htm

WEIDTH_CAMERA, HEIGHT_CAMERA = 640, 480
DEGREE_THRESHOLD = 45

tip_ids = [4, 8, 12, 16, 20]
base_ids = [0, 5, 9, 13, 17]
joint_ids = [3, 6, 10, 14, 18]


cap = cv2.VideoCapture(0)
cap.set(3,WEIDTH_CAMERA)
cap.set(4,HEIGHT_CAMERA)
cur_time = 0
prev_time = 0

detector = htm.HandDetector()


def get_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def is_bent(base,joint,tip):
    v1 = [joint[1] - base[1],joint[2] - base[2]]
    v2 = [tip[1] - joint[1],tip[2] - joint[2]]
    angle = get_angle(v1,v2)
    return angle < DEGREE_THRESHOLD


while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)
    lm_list = detector.find_pos(img,text=True,draw=False)
    if len(lm_list)!=0:
        for finger_index, tip_id in enumerate(tip_ids):
            base_id = base_ids[finger_index]
            joint_id = joint_ids[finger_index]
            if is_bent(lm_list[base_id],lm_list[joint_id],lm_list[tip_id]):
                print('fingers arent bent')
            else:
                print('fingers are bent')


    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, f'FPS{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
