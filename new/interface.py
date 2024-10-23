import tkinter as tk

class CircleGridApp:
    def __init__(self, root, circle_radius=50):
        self.root = root
        self.root.title("Circle Grid")
        
        # Get the screen resolution
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set the resolution to full screen
        self.root.geometry(f"{screen_width}x{screen_height}")

        # Canvas to draw the grid and circles
        self.canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg='white')
        self.canvas.pack()

        # Circle parameters
        self.circle_radius = circle_radius
        self.grid_size = 3
        self.spacing = min(screen_width, screen_height) // (self.grid_size + 1)
        self.centers = []  # To store programmable centers of circles

        # Boolean to toggle the 10th circle
        self.show_10th_circle = True
        self.tenth_circle_center = (1500, 960)  # Programmable center

        # Draw the initial 9 circles
        self.draw_circles()

        # Add button to toggle the 10th circle
        self.toggle_button = tk.Button(root, text="Toggle 10th Circle", command=self.toggle_10th_circle)
        self.toggle_button.pack()

    def draw_circle(self, center, radius, color='black'):
        """Helper function to draw a circle with a center dot"""
        x, y = center
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color)
        # Draw a dot at the center
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color)
        # Display center coordinates as text
        self.canvas.create_text(x, y + radius + 10, text=f"({x}, {y})", fill=color)

    def draw_circles(self):
        """Draw the 3x3 grid of circles"""
        self.canvas.delete("all")  # Clear the canvas before redrawing
        self.centers = []  # Clear the centers list

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = (col + 1) * self.spacing
                y = (row + 1) * self.spacing
                self.centers.append((x, y))
                self.draw_circle((x, y), self.circle_radius)

        # Draw the 10th circle if the boolean is True
        if self.show_10th_circle:
            self.draw_circle(self.tenth_circle_center, self.circle_radius, color='red')

    def toggle_10th_circle(self):
        """Toggle the visibility of the 10th circle"""
        self.show_10th_circle = not self.show_10th_circle
        self.draw_circles()

# Initialize the app with full screen resolution and optional circle radius
if __name__ == "__main__":
    root = tk.Tk()
    app = CircleGridApp(root, circle_radius=20)
    root.mainloop()
