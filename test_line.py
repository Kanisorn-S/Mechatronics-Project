import cv2 as cv
import numpy as np
import joblib
from new.main import calculate_XYZ

model = joblib.load('linear_regression_model.pkl')

# Coordinates for point A and point B
point_A = (200, 300)    # You can change these coordinates
point_B = (400, 300)  # You can change these coordinates

A = np.array([[point_A[0], point_A[1]]])
B = np.array([[point_B[0], point_B[1]]])
# print("A: ", calculate_XYZ(1440, 900, 200, 300))
# print("B: ", calculate_XYZ(1440, 900, 400, 300))

print("A: ", model.predict(A))
print("B: ", model.predict(B))
# Boolean variable to choose line style
is_dashed = False  # Set to True for dashed line, False for solid line

cap = cv.VideoCapture(1)

# # Create a blank white image
# image = np.ones((400, 400, 3), dtype=np.uint8) * 255

# Function to draw a dashed line
def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10):
    if not is_dashed:
        cv.line(img, pt1, pt2, color, thickness)
    else:
        # Calculate the total line length
        line_length = int(np.linalg.norm(np.array(pt2) - np.array(pt1)))
        # Calculate number of dashes
        dashes = line_length // (2 * dash_length)
        
        for i in range(dashes + 1):
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * (i * 2 * dash_length) / line_length),
                int(pt1[1] + (pt2[1] - pt1[1]) * (i * 2 * dash_length) / line_length)
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * ((i * 2 + 1) * dash_length) / line_length),
                int(pt1[1] + (pt2[1] - pt1[1]) * ((i * 2 + 1) * dash_length) / line_length)
            )
            cv.line(img, start, end, color, thickness)


while cap.isOpened():
  ret, frame = cap.read()
  
  # Draw the line from point A to point B
  draw_dashed_line(frame, point_A, point_B, (255, 0, 0), thickness=2)

  # Display the image
  cv.imshow("Line from Point A to Point B", frame)

  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()
