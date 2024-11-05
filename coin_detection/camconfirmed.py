import cv2
import numpy as np
import random

# Initialize variables
coin_data = {}
final_coin_categories = {}
radius_tolerance = 2

# Function to assign color for a radius category
def get_color_for_radius(avg_radius):
    if avg_radius in final_coin_categories:
        return final_coin_categories[avg_radius]['color']
    else:
        new_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        final_coin_categories[avg_radius] = {'color': new_color, 'avg_radius': avg_radius}
        return new_color

# Function to categorize based on recorded radii
def categorize_coin_radius(radius):
    for avg_radius in final_coin_categories:
        if abs(radius - avg_radius) <= radius_tolerance:
            return avg_radius
    return None

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
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
            # Check if we are still recording data or if we are in confirmed state
            if not final_coin_categories:
                # Record radius in coin_data for later averaging
                found = False
                for recorded_radius in coin_data:
                    if abs(recorded_radius - radius) <= radius_tolerance:
                        coin_data[recorded_radius].append(radius)
                        found = True
                        break
                if not found:
                    coin_data[radius] = [radius]
                
                # Display detected radius with a temporary color
                color = (255, 255, 255)  # Temporary color before confirmation
            else:
                # Categorize based on average radius after confirmation
                category_radius = categorize_coin_radius(radius)
                if category_radius is not None:
                    color = final_coin_categories[category_radius]['color']
                else:
                    color = (0, 0, 0)  # Black if it doesn't match any category

            # Draw the circle and the center point
            cv2.circle(frame, (x, y), radius, color, 2)
            cv2.circle(frame, (x, y), 2, color, 3)

            # Display radius and coordinates
            radius_text = f"Radius: {radius} ({x}, {y})"
            cv2.putText(frame, radius_text, (x - 40, y - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the result in real time
    cv2.imshow("Real-Time Coin Detection", frame)

    # Press 'c' to confirm and calculate averages
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c') and not final_coin_categories:
        # Calculate average radius for each unique coin size
        for recorded_radius in coin_data:
            avg_radius = sum(coin_data[recorded_radius]) / len(coin_data[recorded_radius])
            color = get_color_for_radius(avg_radius)
            final_coin_categories[avg_radius] = {'color': color, 'avg_radius': avg_radius}
        print("Confirmation complete. Final coin categories with average radii:", final_coin_categories)

    # Press 'q' to exit the loop
    elif key == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
