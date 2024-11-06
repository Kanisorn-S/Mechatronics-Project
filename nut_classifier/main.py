import cv2 as cv
import numpy as np

# cap = cv.VideoCapture(0)

while True:
  # ret, frame = cap.read()
  frame = cv.imread('./nut.png')
  gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  _, threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)
  
  contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
    print(cnt)
    (x, y, w, h) = cv.boundingRect(cnt)
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
  cv.imshow("Frame", frame)
  cv.imshow("Threshold", threshold)
  if cv.waitKey(1) & 0xFF == ord('q'):
    break
  
# cap.release()
cv.destroyAllWindows()