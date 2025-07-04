import cv2

import HandTrackingClass as htm


class Button:

    def __init__(self,position,width,height,value):
        self.position = position
        self.width = width
        self.height = height
        self.value = value

    def draw(self,image):
        cv2.rectangle(image,self.position,
                      (self.position[0]+self.width,self.position[1]+self.height),
                      (255,255,255),-1)
        cv2.rectangle(image,self.position,
                      (self.position[0]+self.width,self.position[1]+self.height),
                      (0,0,),3)
        cv2.putText(image,self.value,(self.position[0]+30,self.position[1]+30),cv2.FONT_ITALIC,2,
                    (0,0,0),2)

    def click(self,x,y,image):
        if (self.position[0]<x<self.position[0]+self.width and
            self.position[1]<y<self.position[1]+self.height):
                cv2.rectangle(image, self.position,
                              (self.position[0] + self.width, self.position[1] + self.height),
                              (255, 255, 255), -1)
                cv2.rectangle(image, self.position,
                              (self.position[0] + self.width, self.position[1] + self.height),
                              (0, 0,), 3)
                cv2.putText(image, self.value, (self.position[0] + 30, self.position[1] + 30), cv2.FONT_ITALIC, 5,
                            (0, 0, 0), 5)


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = htm.HandDetector()

button_values = [
    ['q','e','t'],
    ['w','r','y'],
]

button_list = []
for x in range(3):
    for y in range(2):
        button = Button((x*100+300,y*100+300),
                        100,100,button_values[y][x])
        button_list.append(button)


while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)
    lm_list = detector.find_pos(img)

    for button in button_list:
        button.draw(img)

    if len(lm_list)!=0:
        length = detector.find_distance(lm_list[8][1:],lm_list[12][1:])
        x,y = lm_list[8][1:]

        if length<50:
            for button in button_list:
                button.click(x,y,img)



    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break