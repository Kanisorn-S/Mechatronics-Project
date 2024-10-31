import cv2
import numpy as np

# Function to apply a mask for detecting skin color
def skin_mask(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask based on the skin color range
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
    
    # Apply some morphological operations to clean the mask
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    
    # Apply Gaussian blur to smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    
    return mask

# Function to determine if the hand is left or right based on contour position
def classify_hand_type(frame, contour):
    # Get the center of the frame
    frame_center_x = frame.shape[1] // 2
    
    # Compute the moments of the contour to find the center of the hand
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        hand_center_x = int(moments["m10"] / moments["m00"])
    else:
        hand_center_x = 0

    # If the hand center is on the left half of the screen, it's a left hand; otherwise, it's a right hand
    if hand_center_x < frame_center_x:
        return "Left"
    else:
        return "Right"

# Function to detect the whole hand and identify the index finger
def detect_hand_and_index_finger(frame, mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_finger_tip = (100, 100)
    
    if contours:
        # Find the largest contour which is likely the hand
        max_contour = max(contours, key=cv2.contourArea)
        
        # Classify if the hand is left or right
        hand_type = classify_hand_type(frame, max_contour)
        
        # Find the convex hull and convexity defects
        hull = cv2.convexHull(max_contour, returnPoints=False)
        if len(hull) > 3:
            # Find convexity defects (points between fingers)
            defects = cv2.convexityDefects(max_contour, hull)
            if defects is not None:
                finger_tips = []
                
                # Find the tip of each finger using the start points of convexity defects
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    
                    # Add the start and end points to the list of potential finger tips
                    finger_tips.append(start)
                    finger_tips.append(end)

                # Remove duplicate points and sort by the x-axis (left to right)
                finger_tips = sorted(list(set(finger_tips)), key=lambda x: x[0])

                # If the hand is classified as left, take the second finger from the right
                # If the hand is classified as right, take the second finger from the left
                if hand_type == "Left":
                    if len(finger_tips) >= 2:
                        index_finger_tip = finger_tips[-2]  # Second from right
                    cv2.circle(frame, index_finger_tip, 10, (0, 255, 0), -1)
                    
                    # Display coordinates in the top-right corner of the screen
                    text = f'Pointing Finger: {index_finger_tip}'
                    cv2.putText(frame, text, (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        
                else:
                    if len(finger_tips) >= 2:
                        index_finger_tip = finger_tips[1]  # Second from left
                    
                    # Draw a circle at the index finger tip
                    cv2.circle(frame, index_finger_tip, 10, (0, 255, 0), -1)
                    
                    # Display coordinates in the top-right corner of the screen
                    text = f'Pointing Finger: {index_finger_tip}'
                    cv2.putText(frame, text, (frame.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw the hand contour
        cv2.drawContours(frame, [max_contour], -1, (255, 0, 0), 2)

        # Display hand type in the top-left corner of the screen
        cv2.putText(frame, f'Hand: {hand_type}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not index_finger_tip:
            index_finger_tip = (100, 100)

    return frame, index_finger_tip

if __name__ == "__main__":
    # Main loop to capture video and detect hand and index finger
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Apply skin mask
        mask = skin_mask(frame)

        # Detect hand and index finger and print the coordinates on the frame
        frame_with_finger, _ = detect_hand_and_index_finger(frame, mask)

        # Display the original frame with index finger detection
        cv2.imshow('Index Finger Detection', frame_with_finger)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()