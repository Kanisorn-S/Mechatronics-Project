import cv2 as cv
import numpy as np
import joblib
import tkinter as tk
import threading
import time
from nut_classifier.model import find_nuts

# Initial coordinates (replace these with your actual dynamic coordinates)
tl = [50, 50]
tr = [150, 50]
br = [150, 150]
bl = [50, 150]

model = joblib.load('linear_regression_model.pkl')

cap = cv.VideoCapture(0)

# Function that runs a while loop to update coordinates
def coordinate_loop():
    global tl, tr, br, bl
    while True:
        # Update coordinates to simulate movement
        ret, frame = cap.read()
        if not ret:
            break

        detected, box = find_nuts(frame, max_size=200)
        # cv.imshow("Result", detected)
        
        X = np.array(box)
        Y = model.predict(X)
        tl = Y[0]
        tr = Y[1]
        br = Y[2]
        bl = Y[3]
        print(Y)
        # Call the function to update the rectangle's position in the GUI thread
        root.after(0, update_rectangle)
        # Small delay to control update speed
        time.sleep(0.1)
    cap.release()
    cv.destroyAllWindows()

# Function to update the rectangle's coordinates on the canvas
def update_rectangle():
    canvas.coords(rectangle, tl[0], tl[1], br[0], br[1])

# Initialize the main window
root = tk.Tk()
root.title("Dynamic Rectangle with Continuous While Loop in Tkinter")

# Get screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
print(screen_width)
print(screen_height)

# Set up the Canvas widget
canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
canvas.pack()

# Create a rectangle on the canvas with initial coordinates
rectangle = canvas.create_rectangle(tl[0], tl[1], br[0], br[1], fill="blue")

# Start the while loop in a separate thread
thread = threading.Thread(target=coordinate_loop, daemon=True)
thread.start()

# Run the Tkinter main loop
root.mainloop()
