import cv2

# Define the crop regions - adjust as needed
crop_regions = {
    "none": None,
    "top": (200, 300, 200, 100),  # top-left x, y, width, height
    "bottom": (250, 150, 200, 100),  # top-left x, y, width, height
    "all": (100, 150, 400, 250),
    "center": (200, 200, 200, 200),
    "machine": (250, 225, 225, 175)
}

# Choose the crop region
crop_choice = "none"  # Change to "none", "top", or "bottom"
crop_region = crop_regions[crop_choice]

# Choose to draw the grid or not
draw_grid = True

# Open the video feed (or image)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # or a path to a video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a grid on the frame
    if draw_grid:
        # Draw a grid on the frame, where each line, horizontal and vertical, are separated by 25 pixels
        # Draw horizontal lines
        for i in range(0, frame.shape[0], 25):
            # Mark every 2 horizontal line with a different color
            if i % 2 == 0:
                cv2.line(frame, (0, i), (frame.shape[1], i), (0, 0, 255), 1)
            else:
                cv2.line(frame, (0, i), (frame.shape[1], i), (0, 255, 0), 1)
            # label the horizontal lines
            cv2.putText(frame, str(i), (10, i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # Draw vertical lines
        for i in range(0, frame.shape[1], 25):
            # Mark every 2 vertical line with a different color
            if i % 2 == 0:
                cv2.line(frame, (i, 0), (i, frame.shape[0]), (0, 0, 255), 1)
            else:
                cv2.line(frame, (i, 0), (i, frame.shape[0]), (0, 255, 0), 1)
            # label the vertical lines
            cv2.putText(frame, str(i), (i, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    
    # Crop the frame to the defined region if cropping is enabled
    if crop_region:
        crop_x, crop_y, crop_width, crop_height = crop_region
        cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        cv2.imshow("Cropped Frame", cropped_frame)
    else:
        cropped_frame = frame

    # Display the original frame with the grid
    cv2.imshow("Original Frame with Grid", frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    print(frame.shape)

cap.release()
cv2.destroyAllWindows()
