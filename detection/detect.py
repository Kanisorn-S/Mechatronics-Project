import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

cap = cv.VideoCapture(1)

while cap.isOpened():
  ret, frame = cap.read()
  blur = cv.GaussianBlur(frame, (21, 21), 3)
  canny = cv.Canny(blur, 50, 100)

  distResol = 1
  angleResol = np.pi/180
  threshold = 150
  lines = cv.HoughLines(canny, distResol, angleResol, threshold)

  k = 3000

  for curLine in lines:
    rho, theta = curLine[0]
    dhat = np.array([[np.cos(theta)], [np.sin(theta)]])
    d = rho * dhat
    lhat = np.array([[-np.sin(theta)], [np.cos(theta)]])
    p1 = d + k * lhat
    p2 = d - k * lhat
    p1 = p1.astype(int)
    p2 = p2.astype(int)
    cv.line(frame, (p1[0][0], p1[1][0]), (p2[0][0], p2[1][0]), (255, 0, 0), 1)

  cv.imshow("Webcam", frame)


  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()