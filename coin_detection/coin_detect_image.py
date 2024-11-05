import cv2
import numpy as np

# Load the image
image = cv2.imread('coins.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur the image to reduce noise
blurred = cv2.GaussianBlur(gray, (11, 11), 0)

# Use HoughCircles to detect coins
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=100, param2=200, minRadius=50, maxRadius=250)

# Define thresholds for different coin sizes (these values need to be fine-tuned for your image)
five_baht_radius_range = (20, 70)  # Example radius range for 5 baht coin
ten_baht_radius_range = (100, 300)   # Example radius range for 10 baht coin

# Check if any circles were detected
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
            pass
            color = (0, 0, 255)  # Red for unrecognized coin size
            label = "Unknown"

        # Draw the circle and the center point
        cv2.circle(image, (x, y), radius, color, 2)
        cv2.circle(image, (x, y), 2, color, 3)

        # Display the label and coordinates
        text = f"{label} ({x}, {y})"
        cv2.putText(image, text, (x - 40, y - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the result
cv2.imshow("Coin Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
