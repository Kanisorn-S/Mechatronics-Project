import cv2 as cv
import numpy as np
from nut_classifier.model import find_nuts
import joblib

model = joblib.load('linear_regression_model.pkl')

cap = cv.VideoCapture(0)

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break
  
  detected, box = find_nuts(frame, max_size=200)
  cv.imshow("Result", detected)

  X = np.array(box)
  Y = model.predict(X)
  
  (tl, tr, br, bl) = Y
  print(Y)
  
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()