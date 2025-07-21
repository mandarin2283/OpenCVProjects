import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation()


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

BG_COLOR = (192, 192, 192)

while True:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = selfie_segmentation.process(img_rgb)

    condition = np.stack(
        (results.segmentation_mask,) * 3, axis=-1) > 0.1

    bg_img = cv2.imread(r"C:\Users\Nikita\PycharmProjects\OpenCVProjects\Pose\fon1.jpg")
    bg_img = cv2.resize(bg_img,(1280,720))
    if bg_img is None: raise Exception
    output_img = np.where(condition, img, bg_img)

    cv2.imshow('result', output_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break