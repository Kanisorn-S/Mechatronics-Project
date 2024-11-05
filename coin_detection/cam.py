import cv2
import numpy as np

# Define thresholds for different coin sizes (these values need to be fine-tuned for your webcam feed)
five_baht_radius_range = (25, 30)  # Example radius range for 5 baht coin
ten_baht_radius_range = (31, 35)   # Example radius range for 10 baht coin

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Use HoughCircles to detect coins
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=20, maxRadius=100)

    # Process detected circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, radius) in circles:
            # Classify coins based on radius
            if five_baht_radius_range[0] <= radius <= five_baht_radius_range[1]:
                color = (255, 0, 0)  # Blue for 5 baht coin
                label = "5 Baht"
            elif ten_baht_radius_range[0] <= radius <= ten_baht_radius_range[1]:
                color = (0, 255, 0)  # Green for 10 baht coin
                label = "10 Baht"
            else:
                color = (0, 0, 255)  # Red for unrecognized coin size
                label = "1 Baht"

            # Draw the circle and the center point
            cv2.circle(frame, (x, y), radius, color, 2)
            cv2.circle(frame, (x, y), 2, color, 3)

            # Display the label and coordinates
            text = f"{label} ({x}, {y})"
            cv2.putText(frame, text, (x - 40, y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the result in real time
    cv2.imshow("Real-Time Coin Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
