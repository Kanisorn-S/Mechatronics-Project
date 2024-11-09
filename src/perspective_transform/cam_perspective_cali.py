import cv2 as cv

cap = cv.VideoCapture(0)

checkpoints = [
  1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1,
]
# Points are [x, y] format
points = [
  [415, 380], [370, 380], [325, 380], [280, 382], [236, 383], [190, 385],
  [420, 350], [373, 350], [325, 350], [280, 350], [232, 350], [185, 350],
  [423, 313], [375, 314], [327, 314], [278, 313], [228, 314], [177, 315],
  [430, 274], [379, 274], [327, 275], [275, 275], [223, 275], [170, 275],
  [437, 230], [383, 230], [328, 231], [273, 231], [217, 230], [161, 230],
  [447, 182], [388, 180], [329, 181], [270, 180], [212, 180], [151, 180],
]

show_grid = False

while cap.isOpened():
  ret, frame = cap.read()
  
  for point in points:
    cv.circle(frame, point, 2, (0, 255, 0), -1)
  
  if show_grid:
    for i in range(0, frame.shape[0], 25):
        # Mark every 2 horizontal line with a different color
        if i % 2 == 0:
            cv.line(frame, (0, i), (frame.shape[1], i), (0, 0, 255), 1)
        else:
            cv.line(frame, (0, i), (frame.shape[1], i), (0, 255, 0), 1)
        # label the horizontal lines
        cv.putText(frame, str(i), (10, i), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Draw vertical lines
    for i in range(0, frame.shape[1], 25):
        # Mark every 2 vertical line with a different color
        if i % 2 == 0:
            cv.line(frame, (i, 0), (i, frame.shape[0]), (0, 0, 255), 1)
        else:
            cv.line(frame, (i, 0), (i, frame.shape[0]), (0, 255, 0), 1)
        # label the vertical lines
        cv.putText(frame, str(i), (i, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
  cv.imshow("Feed", frame)

  if cv.waitKey(1) & 0xFf == ord('q'):
    break
  
cap.release()
cv.destroyAllWindows()