import cv2

# Define the crop region - adjust as needed
crop_x, crop_y, crop_width, crop_height = 200, 300, 200, 100  # top-left x, y, width, height

# Open the video feed (or image)
cap = cv2.VideoCapture(1)  # or a path to a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
