import cv2

from Hands.HandTrackingClass import HandDetector
from HandDistanceClass import HandDistance


WIDTH,HEIGHT = 1280,720

cap = cv2.VideoCapture(0)
cap.set(3,WIDTH)
cap.set(4,HEIGHT)
detector = HandDetector()
distance_measurement = HandDistance(detector)

while True:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=False)
    lm_list = detector.find_pos(img, draw=False)
    if len(lm_list) != 0:
        distance_measurement.display(img, lm_list)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break