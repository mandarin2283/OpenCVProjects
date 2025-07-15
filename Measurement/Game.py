from random import randint
from time import sleep

import cv2

from Hands.HandTrackingClass import HandDetector
from HandDistanceClass import HandDistance


class Point:

    def __init__(self,w,h):
        self.w = w
        self.h = h
        self.color = (0,0,255)

    def get_pos(self):
        position_x = randint(500,self.w - 100)
        position_y = randint(100, self.h - 100)
        return (position_x,position_y)

    def spawn(self,image,pos):
        cv2.circle(image,pos,
                   20,self.color,-1)

    def pressed(self,image,box,dis,pos):
        x,y = pos
        if (box[1]<x<box[0] and box[3]<y<box[2] and dis < 30):
            self.color = (47,255,173)
            self.spawn(image,pos)
            return self.dead(image)

    def dead(self,image):
        self.color = (0,0,255)
        pos = self.get_pos()
        return pos

WIDTH,HEIGHT = 1280,720

cap = cv2.VideoCapture(0)
cap.set(3,WIDTH)
cap.set(4,HEIGHT)
detector = HandDetector()
distance_measurement = HandDistance(detector)

point = Point(WIDTH,HEIGHT)
coords = point.get_pos()
score = 0

while True:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img, draw=False)
    lm_list = detector.find_pos(img, draw=False)
    point.spawn(img,coords)
    if len(lm_list) != 0:
        distance,bound = distance_measurement.display(img, lm_list)
        dead = point.pressed(img,bound,distance,coords)
        if dead:
            coords = dead
            score += 1

    cv2.rectangle(img,(0,0),(500,100),
                  (147,25,255),-1)
    cv2.putText(img,f"score: {score}",(0,100),
                cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),3)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break