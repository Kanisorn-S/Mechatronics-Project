import cv2 as cv
from model import find_nuts

cap = cv.VideoCapture(0)

while cap.isOpened():
  ret, frame = cap.read()
  
  detected, blur, edged = find_nuts(frame)
  cv.imshow("Result", detected)
  cv.imshow("Blur", blur)
  cv.imshow("Edge", edged)
  
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()