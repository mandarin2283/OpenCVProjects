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
            cv2.putText(image, self.value, (self.position[0] + 30, self.position[1] + 45),
                        cv2.FONT_ITALIC,5,(0, 0, 0), 5)
            return True
        else:
            return False


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = htm.HandDetector()

button_values = [
    ['q','w','e','r','t','y','u','i','o','p'],
    ['a','s','d','f','g','h','j','k','l','z'],
    ['x','c','space','v','b','n','m','.',',','?'],
]

my_equation = ''
counter = 0

button_list = []
for x in range(10):
    for y in range(3):
        button = Button((x*100+200,y*100+300),
                        100,100,button_values[y][x])
        button_list.append(button)


while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)

    cv2.rectangle(img,(200,220),(1200,300),(255,255,255),-1)
    cv2.rectangle(img, (200, 220), (1200, 300), (0,0,0), 4)

    for button in button_list:
        button.draw(img)

    lm_list = detector.find_pos(img,draw=True)

    if len(lm_list)!=0:
        length = detector.find_distance(lm_list[8][1:],lm_list[12][1:])
        x,y = lm_list[8][1:]

        if length<50:
            for button in button_list:
                if button.click(x,y,img) and counter==0:
                    my_equation += button.value
                    counter = 1

    if counter != 0:
        counter += 1
        if counter > 10:
            counter = 0

    cv2.putText(img, my_equation,(215,270),cv2.FONT_ITALIC,3,
                (0,0,0),2)

    cv2.imshow('result', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('w'):
        my_equation = ''