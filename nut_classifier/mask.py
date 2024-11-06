import numpy as np
import cv2 

img = cv2.imread('./nuts.png')
blur = cv2.GaussianBlur(img, (5, 5), 0)
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(imgGry, 120, 255, cv2.CHAIN_APPROX_NONE)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for cnt in contours:
  print(cv2.contourArea(cnt))
min_cnt_area = 100
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_cnt_area]

for contour in contours:
  (x, y, w, h) = cv2.boundingRect(contour)
  cv2.rectangle(img, (x, y), (x + w, y + h), (0 ,255, 0), 3)
  

cv2.imshow("Thresh", thresh)
cv2.imshow("Result", img)
cv2.waitKey(0)