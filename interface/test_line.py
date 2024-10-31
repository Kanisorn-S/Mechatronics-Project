import tkinter as tk

# Coordinates for point A and point B
point_A = (321, 467)     # You can change these coordinates
point_B = (1002, 468)   # You can change these coordinates

# Boolean variable to choose line style
is_dashed = False  # Set to True for dashed line, False for solid line

full_screen = False

def toggle_full_screen(event):
        """Toggle between full screen and windowed mode on 'p' key press"""
        full_screen = not full_screen
        root.attributes('-fullscreen', full_screen)

# Create the main window
root = tk.Tk()
root.title("Line from Point A to Point B")
root.bind('<p>', toggle_full_screen)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
# Set up the Canvas
canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg="white")
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
