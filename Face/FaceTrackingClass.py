import cv2
import mediapipe as mp
from mediapipe import tasks
from mediapipe.tasks.python import vision


model_path = 'face_landmarker.task'

base_opt = tasks.BaseOptions(model_asset_path=model_path)
face_landmarker = vision.FaceLandmarker
face_land_opt = vision.FaceLandmarkerOptions
face_land_res = vision.FaceLandmarkerResult
vision_running_mode = vision.RunningMode


def return_result(result: face_land_res, output_image: mp.Image, timestamp_ms: int):
    return 'face landmarker result: {}'.format(result)


options = face_land_opt(base_options=base_opt,
                        running_mode=vision_running_mode.LIVE_STREAM,
                        result_callback=return_result
                        )
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while True:
    suc, img = cap.read()
    img = cv2.flip(img, 1)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=img)

    timems = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detector.detect_async(mp_img,timems)

    cv2.imshow('result', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break