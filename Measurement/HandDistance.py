import cv2
import numpy as np

from Hands.HandTrackingClass import HandDetector

WIDTH,HEIGHT = 1280,720

cap = cv2.VideoCapture(0)
cap.set(3,WIDTH)
cap.set(4,HEIGHT)
detector = HandDetector()

x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
coff = np.polyfit(x, y, 2)

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    h,w,_ = img.shape
    img = detector.find_hands(img,draw=False)
    lm_list = detector.find_pos(img,draw=False)
    if len(lm_list)!=0:
        handedness = detector.find_handedness(img,lm_list,draw=False)
        bound_box = detector.bound_box(img, lm_list)
        distance = int(detector.find_distance(lm_list[5][1:],lm_list[17][1:]))
        A, B, C = coff
        distance_cm =  A * distance ** 2 + B * distance + C
        x1,y1 = bound_box[1],bound_box[3]
        x2,y2 = bound_box[0],y1-50
        cv2.rectangle(img,(x1,y1),(x2,y2),
                      (0,0,255),-1)
        cv2.putText(img,f"{int(distance_cm)} cm",
                    (x1,y1),cv2.FONT_HERSHEY_PLAIN,
                    3,(255,255,255),3)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break