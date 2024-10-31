import cv2 as cv
from finger_detection.finger_detect import skin_mask, detect_hand_and_index_finger
from new.main import calculate_XYZ
from new.test_move import move_cursor
import pyautogui
from depth_estimation.main import Midas, ModelType

pyautogui.FAILSAFE = False

midasObj = Midas(ModelType.MIDAS_SMALL)
midasObj.useCUDA()
midasObj.transform()

cap = cv.VideoCapture(1)

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break
  
  # find index finger coordinates in (u, v)
  frame = cv.flip(frame, 1)
  mask = skin_mask(frame)
  frame_with_finger, index_finger_pos = detect_hand_and_index_finger(frame, mask)
  u, v = index_finger_pos

  # MiDaS Depth Estimation
  depth = midasObj.predict_point(frame, (v, u))
  print(depth)
  # depthMap = midasObj.predict(frame)

  # cv.putText(frame_with_finger, "Depth: " + str(depth), (100, 100), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)
  
  
  # convert (u, v) to (x, y)
  x, y, z = calculate_XYZ(100, 100, u, v)
  
  # Check Depth
  if (depth < 180):
    # invert x, y for projector
    x = 1440 - x
    # update cursor position
    move_cursor(x * 2, y * 2)

  # display cv
  cv.imshow("Index Finger Detection", frame_with_finger)
  
  if cv.waitKey(1) & 0xFF == ord('q'):
    break
  
cap.release()
cv.destroyAllWindows()
  
  