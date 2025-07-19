import cv2
import mediapipe as mp


face_mesh = mp.solutions.face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while True:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for lm in face_landmarks.landmark:
                h,w,_ = img.shape
                cx,cy = int(lm.x * w),int(lm.y * h)
                cv2.circle(img,(cx,cy),7,
                           (200,10,75),-1)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break