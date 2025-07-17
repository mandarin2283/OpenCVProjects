import cv2
import numpy as np


def get_contours(image,threshold=[100,100],min_area=1000,
                 filter_threshold=0,draw=True,show=False):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray,[5,5],1)
    image_canny = cv2.Canny(image_blur,threshold[0],threshold[1])
    kernel = np.ones((5,5))
    image_dil = cv2.dilate(image_canny,kernel,iterations=3)
    image_threshold = cv2.erode(image_dil,kernel,iterations=2)
    if show:
        cv2.imshow('rrr',image_threshold)
    contours,_ = cv2.findContours(image_threshold,cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

    final_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area>min_area:
            perimeter = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02*perimeter,True)
            bbox = cv2.boundingRect(approx)
            if filter_threshold > 0:
                if filter_threshold == len(approx):
                    final_contours.append([len(approx),area,approx,bbox,i])
            else:
                final_contours.append([len(approx), area, approx, bbox, i])
    final_contours = sorted(final_contours,key=lambda x: x[1],reverse=True)
    if draw:
        for contour in final_contours:
            cv2.drawContours(image,contour[4],-1,(0,0,255),3)
    return image,final_contours


def reorder(points):
    new_points = np.zeros_like(points)
    points = points.reshape((4,2))
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points,axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points



def warp_image(image,points,w,h,pad=20):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    image_warp = cv2.warpPerspective(image,matrix,(w,h))
    image_warp = image_warp[pad:image_warp.shape[0]-pad,pad:image_warp.shape[1]-pad]
    return image_warp