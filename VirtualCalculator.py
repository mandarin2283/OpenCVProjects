import cv2
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    
    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break