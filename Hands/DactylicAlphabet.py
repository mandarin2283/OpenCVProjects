import cv2
import numpy as np

from Hands.HandTrackingClass import HandDetector

WIDTH_CAMERA, HEIGHT_CAMERA = 640, 480


class Letter:
    def __init__(self,name,direction_list,bent_fingers_list):
        self.name = name
        self.direction_list = direction_list
        self.bent_fingers_list = bent_fingers_list

    def compare(self,conditions):
        test_list = []

        if any([x==conditions[0] for x in self.direction_list]):
            test_list.append(1)
            #print()
        else: return False
        if all([a.count(conditions[1][i])
                        for i,a in enumerate(self.bent_fingers_list)]):
            test_list.append(1)
        else: return False
        return self.name


cap = cv2.VideoCapture(0)
cap.set(3,WIDTH_CAMERA)
cap.set(4,HEIGHT_CAMERA)

detector = HandDetector()

letter_list = []
letter_0 = Letter('а',['right','left'],[[1],[1],[1],[1],[1]])
letter_2 = Letter('в', ['up'],[[0,1],[0],[0],[0],[0]])
letter_list.extend([letter_0,letter_2])

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)
    lm_list = detector.find_pos(img)
    if len(lm_list)!=0:
        dot = detector.main_dot(lm_list)
        dir_hand = detector.dir_hand(dot,lm_list)
        bent_list = detector.is_bent(lm_list)
        condition_list = [dir_hand,bent_list] # ['up', [0, 0, 0, 0, 0]]

        for letter in letter_list:
            res = letter.compare(condition_list)
            if res: print(res)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
