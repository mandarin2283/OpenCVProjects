import cv2
import numpy as np

from Hands import HandTrackingClass as htm

WIDTH_CAMERA, HEIGHT_CAMERA = 640, 480
DEGREE_THRESHOLD = 45

tip_ids = [4, 8, 12, 16, 20]
base_ids = [0, 5, 9, 13, 17]
joint_ids = [3, 6, 10, 14, 18]


cap = cv2.VideoCapture(0)
cap.set(3,WIDTH_CAMERA)
cap.set(4,HEIGHT_CAMERA)

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


def is_bent(lm_list):
    bent_list = []
    for finger_index,tip_id in enumerate(tip_ids):
        base_id = base_ids[finger_index]
        joint_id = joint_ids[finger_index]

        v1 = [lm_list[joint_id][1]-lm_list[base_id][1],
              lm_list[joint_id][2]-lm_list[base_id][2]]
        v2 = [lm_list[tip_id][1]-lm_list[joint_id][1],
              lm_list[tip_id][2]-lm_list[joint_id][2]]
        if get_angle(v1,v2) < DEGREE_THRESHOLD:
            bent_list.append(0)
        else:
            bent_list.append(1)
    return bent_list


while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)
    lm_list = detector.find_pos(img,text=True,draw=False)
    if len(lm_list)!=0:
        fingers_bent = is_bent(lm_list)
        if all(fingers_bent) == 1:
            print('a')



    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
