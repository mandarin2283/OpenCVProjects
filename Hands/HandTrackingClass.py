import time
import math

import cv2
import mediapipe as mp
import numpy as np



class HandDetector():

    def __init__(self,mode=False,max_hands=2,complexity=1,
                 min_detect=0.5,min_track=0.5):
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
                    self.mpDraw.draw_landmarks(img, hand_landmarks,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def find_pos(self,img,hand_no=0,draw=True,text=False):
        lm_list = []

        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]

            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id,cx,cy])
                if text:
                    cv2.putText(img,str(id),(cx,cy),cv2.FONT_HERSHEY_PLAIN,
                                1.5,(0,255,0),1)
                if draw:
                    cv2.circle(img,(cx,cy),10,(50,255,50),-1)

        return lm_list

    def is_bent(self,lm_list):
        tip_ids = [4, 8, 12, 16, 20]
        base_ids = [0, 5, 9, 13, 17]
        joint_ids = [3, 6, 10, 14, 18]
        DEGREE_THRESHOLD = 45
        bent_list = []
        for finger_index, tip_id in enumerate(tip_ids):
            base_id = base_ids[finger_index]
            joint_id = joint_ids[finger_index]

            v1 = [lm_list[joint_id][1] - lm_list[base_id][1],
                  lm_list[joint_id][2] - lm_list[base_id][2]]
            v2 = [lm_list[tip_id][1] - lm_list[joint_id][1],
                  lm_list[tip_id][2] - lm_list[joint_id][2]]
            if self.get_angle(v1, v2) < DEGREE_THRESHOLD:
                bent_list.append(0)
            else:
                bent_list.append(1)
        return bent_list

    def find_distance(self,p1,p2):
        x1,y1 = p1
        x2,y2 = p2
        length = math.hypot(x2-x1,y2-y1)
        return length

    def main_dot(self,lm_list):
        x1, y1 = lm_list[0][1:]
        x2, y2 = lm_list[9][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return cx, cy

    def dir_hand(self,dot, lm_list):
        dir_dot = lm_list[9][1:]
        vector = np.array(dir_dot) - np.array(dot)
        vx, vy = vector
        module = abs(vector)
        if module[0] > module[1]:
            if vx > 0:
                return 'right'
            elif vx < 0:
                return 'left'
        elif module[0] < module[1]:
            if vy > 0:
                return 'down'
            elif vy < 0:
                return 'up'

    @staticmethod
    def get_angle(v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cosine_angle = dot_product / (norm_v1 * norm_v2)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)


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