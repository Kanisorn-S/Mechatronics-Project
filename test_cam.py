import cv2

# Define the crop region - adjust as needed
# Crop for top camera
# crop_x, crop_y, crop_width, crop_height = 200, 300, 200, 100  # top-left x, y, width, height

# Crop for bottom camera
crop_x, crop_y, crop_width, crop_height = 250, 150, 200, 100  # top-left x, y, width, height

# Open the video feed (or image)
cap = cv2.VideoCapture(1)  # or a path to a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # write a code that will draw a grid on the frame, where each line, horizontal and vertical, are separated by 25 pixels
    # draw horizontal lines
    for i in range(0, frame.shape[0], 25):
        cv2.line(frame, (0, i), (frame.shape[1], i), (0, 255, 0), 1)
    # draw vertical lines
    for i in range(0, frame.shape[1], 25):
        cv2.line(frame, (i, 0), (i, frame.shape[0]), (0, 255, 0), 1)
        
        
    print(frame.shape)
    # Crop the frame to the defined region
    cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

    # Display the original frame with the translated contours
    cv2.imshow("Original Frame with Edge Contours", frame)
    
    # Display cropped frame (optional)
    cv2.imshow("Cropped Frame", cropped_frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
