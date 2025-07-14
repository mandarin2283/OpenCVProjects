import cv2

import HandTrackingClass as htm


class Shape:

    def __init__(self,center_position,size):
        self.size = size
        self.center_position = center_position

    def draw(self,image):
        alpha = 0.5
        beta = 1 - alpha
        over = image.copy()
        cv2.rectangle(over, (self.center_position[0] - self.size[0] // 2,
                              self.center_position[1] - self.size[1] // 2),
                      (self.center_position[0] + self.size[0] // 2,
                       self.center_position[1] + self.size[1] // 2),
                      (200,100,80),-1)
        cv2.addWeighted(over,alpha,image,beta,0.0,image)


    def update(self,cursor):
        cx,cy = self.center_position
        width,height = self.size
        if (cx - width//2<cursor[0]< cx + width//2 and
            cx - height//2<cursor[1]< cx + height//2 ):
            self.center_position = cursor


cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

shape = Shape((200,200),(150,150))

while True:
    suc, img = cap.read()
    img = cv2.flip(img,1)
    img = detector.find_hands(img)

    shape.draw(img)

    lm_list = detector.find_pos(img)
    if len(lm_list)!=0:
        length = detector.find_distance(lm_list[8][1:],lm_list[12][1:])
        cursor = lm_list[8][1:]
        if length<50:
            shape.update(cursor)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
