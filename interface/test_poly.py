import tkinter as tk

# Coordinates for the corners of the rectangle
point_A = (492, 280)    # Top left corner
point_B = (491, 466)   # Top right corner
point_C = (661, 467)  # Bottom right corner
point_D = (662, 280)   # Bottom left corner

# Create the main window
root = tk.Tk()
root.title("Rectangle with Corners A, B, C, and D")

# Set up the Canvas
canvas = tk.Canvas(root, width=400, height=400, bg="white")
canvas.pack()

# Draw the rectangle using points A, B, C, and D
canvas.create_polygon(
    point_A[0], point_A[1],
    point_B[0], point_B[1],
    point_C[0], point_C[1],
    point_D[0], point_D[1],
    outline="blue", fill="", width=2
)

# Run the application
root.mainloop()
