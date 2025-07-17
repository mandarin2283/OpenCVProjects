import cv2
import utils

path = '5460877220252875051.jpg'

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
                                                        min_area=3000,threshold=[30,30],show=True)
        #cv2.imshow('res',img_cont2)

    #cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break