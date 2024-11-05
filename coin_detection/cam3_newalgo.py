import cv2
import numpy as np
import random

# Initialize a list to store unique coin sizes and colors for each size
coin_sizes = []
coin_colors = {}

# Define a tolerance for grouping similar coin sizes
radius_tolerance = 2

# Function to assign a color for each unique coin size
def get_color_for_radius(radius):
    for registered_radius in coin_sizes:
        if abs(radius - registered_radius) <= radius_tolerance:
            return coin_colors[registered_radius]
    
    # If it's a new size, assign a random color
    coin_sizes.append(radius)
    new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    coin_colors[radius] = new_color
    return new_color

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set higher resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply GaussianBlur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Use HoughCircles to detect coins
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.1, minDist=50,
                               param1=100, param2=30, minRadius=20, maxRadius=100)

    # Process detected circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, radius) in circles:
            # Get the color for this coin size, assigning a new color if it's a new size
            color = get_color_for_radius(radius)

            # Use contours for a smoother boundary
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), radius, 255, -1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)

            # Display radius and coordinates
            radius_text = f"Radius: {radius} ({x}, {y})"
            cv2.putText(frame, radius_text, (x - 40, y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the result in real time
    cv2.imshow("Real-Time Coin Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
