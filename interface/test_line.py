import tkinter as tk

# Coordinates for point A and point B
point_A = (50, 50)     # You can change these coordinates
point_B = (200, 200)   # You can change these coordinates

# Boolean variable to choose line style
is_dashed = False  # Set to True for dashed line, False for solid line

# Create the main window
root = tk.Tk()
root.title("Line from Point A to Point B")

# Set up the Canvas
canvas = tk.Canvas(root, width=400, height=400, bg="white")
canvas.pack()

# Determine the line style based on the is_dashed variable
line_style = (5, 5) if is_dashed else None  # Dashed if True, solid if False

# Draw a line from point A to point B
canvas.create_line(
    point_A[0], point_A[1], point_B[0], point_B[1],
    fill="blue", width=2, dash=line_style
)

# Run the application
root.mainloop()
