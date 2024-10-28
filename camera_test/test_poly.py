import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# Coordinates for the corners of the rectangle
point_A = (50, 50)    # Top left corner
point_B = (200, 50)   # Top right corner
point_C = (200, 150)  # Bottom right corner
point_D = (50, 150)   # Bottom left corner

# # Create a blank white image
# image = np.ones((400, 400, 3), dtype=np.uint8) * 255

while cap.isOpened():
  ret, frame = cap.read()
  
  # Draw the rectangle by connecting points A, B, C, and D
  cv.line(frame, point_A, point_B, (0, 0, 255), 2)
  cv.line(frame, point_B, point_C, (0, 0, 255), 2)
  cv.line(frame, point_C, point_D, (0, 0, 255), 2)
  cv.line(frame, point_D, point_A, (0, 0, 255), 2)

  # Display the image
  cv.imshow("Rectangle with Corners A, B, C, and D", frame)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break


cap.release()
cv.destroyAllWindows()
