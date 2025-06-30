import cv2
from cvzone.HandTrackingModule import HandDetector


class Button:

    def __init__(self,position,width,height,value):
        self.position = position
        self.width = width
        self.height = height
        self.value = value

    def draw(self,image):
        cv2.rectangle(image, self.position,
                      (self.position[0]+self.width,self.position[1]+self.height),
                      (255, 255, 255), -1)
        cv2.rectangle(image, self.position,
                      (self.position[0]+self.width,self.position[1]+self.height),
                      (50, 50, 50), 3)
        cv2.putText(image, self.value, (self.position[0] + 40, self.position[1] + 60),
                    cv2.FONT_HERSHEY_PLAIN,
                    2, (50, 50, 50), 2)


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = HandDetector(detectionCon=0.8,maxHands=1)

button_values = [['7','4','1','0'],
                 ['8','5','2','/'],
                 ['9','6','3','.'],
                 ['*','-','+','='],
                 ]
button_list = []
for x in range(4):
    for y in range(4):
        xpos = x*100 + 800
        ypos = y*100 + 150
        button_list.append(Button((xpos, ypos), 100, 100, button_values[x][y]))

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)

    hands,img = detector.findHands(img,flipType=False)

    for button in button_list:
        button.draw(img)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break