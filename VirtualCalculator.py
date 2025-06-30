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

    def click(self,x,y,image):
        if (self.position[0]<x<self.position[0]+self.width and
            self.position[1]<y<self.position[1]+self.height):
            cv2.rectangle(image, self.position,
                          (self.position[0] + self.width, self.position[1] + self.height),
                          (255, 255, 255), -1)
            cv2.rectangle(image, self.position,
                          (self.position[0] + self.width, self.position[1] + self.height),
                          (50, 50, 50), 3)
            cv2.putText(image, self.value, (self.position[0] + 20, self.position[1] + 70),
                        cv2.FONT_HERSHEY_PLAIN,
                        5, (0, 0, 0), 5)
            return True
        else:
            return False


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
detector = HandDetector(detectionCon=0.8,maxHands=1)

button_values = [['7','8','9','*'],
                 ['4','5','6','-'],
                 ['1','2','3','+'],
                 ['0','/','.','='],
                 ]
button_list = []
for x in range(4):
    for y in range(4):
        xpos = x*100 + 800
        ypos = y*100 + 150
        button_list.append(Button((xpos, ypos), 100, 100,
                                  button_values[y][x]))

my_equation = ''
counter = 0


while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)

    hands,img = detector.findHands(img,flipType=False)

    cv2.rectangle(img,(800,70),(1200,170),
                  (255,255,255),-1)
    cv2.rectangle(img,(800,70),(1200,170),
                  (50,50,50),3)
    for button in button_list:
        button.draw(img)

    if hands:
        lm_list = hands[0]['lmList']
        length,info,img = detector.findDistance(lm_list[8][:2],lm_list[12][:2],img)
        x,y,z = lm_list[8]
        if length<50:
            for i,button in enumerate(button_list):
                if button.click(x,y,img) and counter==0:
                    my_value = button_values[int(i%4)][int(i/4)]
                    if my_value=='=':
                        my_equation = str(eval(my_equation))
                    else:
                        my_equation += my_value
                    counter = 1
    if counter != 0:
        counter += 1
        if counter>10:
            counter = 0

    cv2.putText(img, my_equation, (810,120),
                cv2.FONT_HERSHEY_PLAIN,
                3, (50, 50, 50), 3)

    cv2.imshow('result', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('w'):
        my_equation = ''