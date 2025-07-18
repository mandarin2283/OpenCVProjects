import cv2
import utils

path = 'cards.jpg'

cap = cv2.VideoCapture(1)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)

scale = 3
w = 210*scale
h = 270*scale


while True:
    #suc,img = cap.read()
    img = cv2.imread(path)
    #img = cv2.flip(img,1)
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    img_cont,final_contours =  utils.get_contours(img,
                                             filter_threshold=4,min_area=50000)
    if len(final_contours)!=0 :
        biggest = final_contours[0][2]
        img_warp = utils.warp_image(img,biggest,w,h)
        img_cont2, final_contours2 = utils.get_contours(img_warp,filter_threshold=4,
                                                        min_area=3000,threshold=[30,30])

        if len(final_contours2)!=0:
            for dot in final_contours2:
                cv2.polylines(img_cont2,[dot[2]],True,(0,0,255),2)

                new_points = utils.reorder(dot[2])
                width = round((utils.find_distance(new_points[0][0]//scale,new_points[[1][0]]//scale)/10),1)
                height = round((utils.find_distance(new_points[0][0]//scale,new_points[[2][0]]//scale)/10),1)

        cv2.arrowedLine(img_cont2,(new_points[0][0][0],new_points[0][0][1]),
                       (new_points[1][0][0],new_points[1][0][1]),
                      (0,0,255),3,8,0,0.05)
        cv2.arrowedLine(img_cont2,(new_points[0][0][0],new_points[0][0][1]),
                     (new_points[2][0][0],new_points[2][0][1]),
                    (0,0,255),3,8,0,0.05)
        x,y,w_dot,h_dot = dot[3]
        cv2.putText(img_cont2, '{}cm'.format(width), (x + 30, y - 10),
                  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                 (255, 0, 255), 2)
        cv2.putText(img_cont2, '{}cm'.format(height), (x - 70, y + h_dot // 2),
                  cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                 (255, 0, 255), 2)
        cv2.imshow('res', img_cont2)

    #cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break