import os 
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(1)

savedir = "camera_info"
newcam_mtx=np.load(os.path.join(savedir, 'newcam_mtx.npy'))

cx=newcam_mtx[0,2]
cy=newcam_mtx[1,2]
fx=newcam_mtx[0,0]

while cap.isOpened():
  ret, frame = cap.read()
  
  cv.circle(frame, (303-50, 237), 2, (255, 0, 0), -1)
  cv.circle(frame, (367-50, 237), 2, (255, 0, 0), -1)
  cv.circle(frame, (430-50, 235), 2, (255, 0, 0), -1)
  cv.circle(frame, (303-50, 294), 2, (255, 0, 0), -1)
  cv.circle(frame, (367-50, 293), 2, (255, 0, 0), -1)
  cv.circle(frame, (433-50, 293), 2, (255, 0, 0), -1)
  cv.circle(frame, (298-50, 357), 2, (255, 0, 0), -1)
  cv.circle(frame, (367-50, 357), 2, (255, 0, 0), -1)
  cv.circle(frame, (437-50, 357), 2, (255, 0, 0), -1)
  h, w, _ = frame.shape
  # for i in range(10):
  #   cv.line(frame, (i  * 50, 0), (i * 50, h), (255, 0, 0), 2)
  #   cv.line(frame, (0, i * 50), (w, i * 50), (255, 0, 0), 2)
  cv.imshow("Webcam", frame)
  
  if cv.waitKey(1) & 0xFF == ord('q'):
    break
  
cap.release()
cv.destroyAllWindows()