import cv2
import numpy as np


img1 = cv2.imread('images/ps1.jpg',0)
img2 = cv2.imread('images/ps2.jpg',0)
img1 = cv2.resize(img1,(500,400))
img2 = cv2.resize(img2,(500,400))

orb = cv2.ORB_create(nfeatures=1000)

kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

# cv2.imshow('e',img1)
# cv2.imshow('e',img1)
cv2.imshow('e',img3)
cv2.waitKey(0)