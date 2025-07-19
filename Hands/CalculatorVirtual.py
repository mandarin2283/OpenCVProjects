import cv2
import HandTrackingClass as htm


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





def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    detector = htm.HandDetector()

    button_values = [['7', '8', '9', '*'],
                     ['4', '5', '6', '-'],
                     ['1', '2', '3', '+'],
                     ['0', '/', '.', '='],
                     ]
    button_list = []
    for x in range(4):
        for y in range(4):
            xpos = x * 100 + 600
            ypos = y * 100 + 100
            button_list.append(Button((xpos, ypos), 100, 100,
                                      button_values[y][x]))

    my_equation = ''
    counter = 0

    while True:
        suc, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.find_hands(img)

        cv2.rectangle(img, (600, 50), (1000, 170),
                      (255, 255, 255), -1)
        cv2.rectangle(img, (600, 50), (1000, 170),
                      (50, 50, 50), 3)
        for button in button_list:
            button.draw(img)

        lm_list = detector.find_pos(img)

        if len(lm_list) != 0:
            length = detector.find_distance(lm_list[8][1:], lm_list[12][1:])
            x, y = lm_list[8][1:]
            if length < 45:
                for i, button in enumerate(button_list):
                    if button.click(x, y, img) and counter == 0:
                        my_value = button.value
                        if my_value == '=':
                            my_equation = str(eval(my_equation))
                        else:
                            my_equation += my_value
                        counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0

        cv2.putText(img, my_equation, (610, 90),
                    cv2.FONT_HERSHEY_PLAIN,
                    3, (50, 50, 50), 3)

        cv2.imshow('result', img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            my_equation = ''


if __name__ == '__main__':
    main()