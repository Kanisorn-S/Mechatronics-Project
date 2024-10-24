import cv2 as cv
from finger_detection.finger_detect import skin_mask, detect_hand_and_index_finger
from new.main import calculate_XYZ
from new.test_move import move_cursor
import pyautogui

pyautogui.FAILSAFE = False


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
  
  # convert (u, v) to (x, y)
  x, y, z = calculate_XYZ(u, v)
  
  # update cursor position
  move_cursor(x * 2, y * 2)

  # display cv
  cv.imshow("Index Finger Detection", frame_with_finger)
  
  if cv.waitKey(1) & 0xFF == ord('q'):
    break
  
cap.release()
cv.destroyAllWindows()
  
  