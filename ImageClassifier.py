import os

import cv2
import numpy as np


orb = cv2.ORB_create(nfeatures=1000)

path = 'images'
images = []
class_names = []
my_list = os.listdir(path)

for my_class in my_list:
    current_image = cv2.imread(f'{path}/{my_class}',0)
    images.append(current_image)
    class_names.append(os.path.splitext(my_class)[0])


def find_descriptor(images):
    des_list = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        des_list.append(des)
    return des_list


def find_id(img,des_list):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    match_list = []
    for des in des_list:
        matches = bf.knnMatch(des, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        match_list.append(len(good))
    print(match_list)


descriptors = find_descriptor(images)

cap = cv2.VideoCapture(0)

while True:

    suc,image  = cap.read()
    image_original = image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    find_id(image,descriptors)

    cv2.imshow('result',image_original)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break