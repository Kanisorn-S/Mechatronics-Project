import cv2

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():

  ret, frame = cap.read()

  k = cv2.waitKey(5)

  if k == ord('q'):
    break
  elif k == ord('s'):
    cv2.imwrite('images/img' + str(num) + '.png', frame)
    print("image saved!")
    num += 1
  
  cv2.imshow('Camera', frame)

cap.release()
cv2.destroyAllWindows()