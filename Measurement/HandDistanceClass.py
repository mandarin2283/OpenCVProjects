import numpy as np
import cv2

from Hands.HandTrackingClass import HandDetector


class HandDistance:
    x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
    y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    def __init__(self,detector):
        self.detector = detector

    def get_distance(self,lm_list):
        coff = np.polyfit(self.x, self.y, 2)
        distance = int(self.detector.find_distance(lm_list[5][1:], lm_list[17][1:]))
        A, B, C = coff
        distance_cm = A * distance ** 2 + B * distance + C
        return distance_cm

    def display(self,image,lm_list):
        bound_box = self.detector.bound_box(image, lm_list)
        x1, y1 = bound_box[1], bound_box[3]
        x2, y2 = bound_box[0], y1 - 50
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 0, 255), -1)
        cv2.putText(image, f"{int(self.get_distance(lm_list))} cm",
                    (x1, y1), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 255, 255), 3)

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    distance_measurement = HandDistance(detector)

    while True:
        suc, img = cap.read()
        img = cv2.flip(img,1)
        img = detector.find_hands(img,draw=False)
        lm_list = detector.find_pos(img,draw=False)
        if len(lm_list)!=0:
            distance_measurement.display(img,lm_list)

        cv2.imshow('result', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()