import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Vertical Line Example")

# Create a Canvas widget
canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
canvas.pack()

# Define the position of the vertical line (e.g., 100 pixels from the left edge)
x_position = 100
canvas_height = root.winfo_screenheight()  # Height of the canvas

# Draw the vertical line
# x1, y1, x2, y2 (start and end coordinates of the line)
canvas.create_line(x_position, 0, x_position, canvas_height, fill="black", width=2)

# Run the Tkinter event loop
root.mainloop()
