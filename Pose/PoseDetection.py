import mediapipe as mp
import cv2


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
cap = cv2.VideoCapture(0)

while True:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = holistic.process(img_rgb)
    mp_drawing.draw_landmarks(
        img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break