import numpy as np
import cv2 as cv 
import time 
import os

# Set Window for display
cv.namedWindow("Image Feed")
cv.moveWindow("Image Feed", 159, -25)

# Creating a camera object for webcam
cap = cv.VideoCapture(0)

# Set resolution and framerates according to camera specs
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 3280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 2464)
# cap.set(cv.CAP_PROP_FPS, 90)

prev_frame_time = time.time()

cal_image_count = 0
frame_count = 0

while cap.isOpened():

    ret, frame = cap.read()

    frame_count += 1

    if frame_count == 30:
        img_path = os.path.join("calibration_images", "cal_image_" + str(cal_image_count) + ".jpg")
        cv.imwrite(img_path, frame)
        cal_image_count += 1
        frame_count = 0

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv.putText(frame, "FPS " + str(int(fps)), (10, 40), cv.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 2, cv.LINE_AA)
    cv.imshow("Image feed", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()