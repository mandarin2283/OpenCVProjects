import cv2
import mediapipe as mp
import time


class HandDetector():

    def __init__(self,mode=False,max_hands=2,complexity=1,min_detect=0.5,min_track=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.min_detect = min_detect
        self.min_track = min_track

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode,max_hands,
                                        complexity,min_detect,min_track)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_pos(self,img,hand_no=0,draw=True):
        lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id,cx,cy])
                if draw:
                    cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),1)

        return lm_list



def main():
    prev_time = 0
    cur_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        suc, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_pos(img)

        cur_time = time.time()
        fps = 1/(cur_time-prev_time)
        prev_time = cur_time
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,
        (255,0,255),3)

        cv2.imshow('result', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()