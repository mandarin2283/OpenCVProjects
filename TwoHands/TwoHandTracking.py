import cv2

from Hands.HandTrackingClass import HandDetector


cap = cv2.VideoCapture(0)
detector = HandDetector()

while True:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img)
    all_landmarks = detector.find_pos_two(img)
    if len(all_landmarks) > 21:
        point1 = all_landmarks[8][1:]
        point2 = all_landmarks[29][1:]
        cv2.line(img,point1,point2,(0,0,255),10)
        distance = detector.find_distance(point1,point2)
        print(distance)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break