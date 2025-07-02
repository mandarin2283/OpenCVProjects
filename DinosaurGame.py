import cv2

import HandTrackingClass as htm

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

pos_y = 400

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)
    lm_list = detector.find_pos(img)

    if len(lm_list)!=0:
        length = detector.find_distance(lm_list[4][1:],lm_list[8][1:])

        if length<20:
            pos_y -= 10

    cv2.rectangle(img,(100,pos_y-200),(200,pos_y),(0,255,0),-1)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break