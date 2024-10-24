import tkinter as tk

class CircleGridApp:
    def __init__(self, root, circle_radius=50):
        self.root = root
        self.root.title("Circle Grid")

        # Get the screen resolution
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()

        # Set the initial resolution to full screen size
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")

        # Canvas to draw the grid and circles
        self.canvas = tk.Canvas(root, width=self.screen_width, height=self.screen_height, bg='white')
        self.canvas.pack()

        # Circle parameters
        self.circle_radius = circle_radius
        self.grid_size = 3
        self.spacing = min(self.screen_width, self.screen_height) // (self.grid_size + 1)
        self.centers = []  # To store programmable centers of circles

        # Boolean to toggle the 10th circle manually in code
        self.show_10th_circle = False
        # self.tenth_circle_center = (666, 473)  # Programmable center
        self.tenth_circle_center = (1018, 810)  # Programmable center

        # Draw the initial 9 circles
        self.draw_circles()

        # Bind 'p' key for full screen toggle
        self.root.bind('<p>', self.toggle_full_screen)
        self.root.bind('<Escape>', self.close)

        # Track full screen state
        self.full_screen = False

    def draw_line(self):
        x_position = 100
        self.canvas.create_line(x_position, 0, x_position, self.screen_height, fill="black", width=2)
        
    def draw_circle(self, center, radius, color='black'):
        """Helper function to draw a circle with a center dot"""
        x, y = center
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color)
        # Draw a dot at the center
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color)
        # Display center coordinates as text
        self.canvas.create_text(x, y + radius + 10, text=f"({x}, {y})", fill=color)

    def draw_circles(self):
        """Draw the 3x3 grid of circles, centered in the screen"""
        self.canvas.delete("all")  # Clear the canvas before redrawing
        self.centers = []  # Clear the centers list

        # Calculate the center offset to place the grid in the middle of the screen
        start_x = (self.screen_width - (self.spacing * (self.grid_size - 1))) // 2
        start_y = (self.screen_height - (self.spacing * (self.grid_size - 1))) // 2

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = start_x + col * self.spacing
                y = start_y + row * self.spacing
                self.centers.append((x, y))
                self.draw_circle((x, y), self.circle_radius)

        # Draw the 10th circle if the boolean is True
        if self.show_10th_circle:
            self.draw_circle(self.tenth_circle_center, self.circle_radius, color='red')

    def toggle_full_screen(self, event):
        """Toggle between full screen and windowed mode on 'p' key press"""
        self.full_screen = not self.full_screen
        self.root.attributes('-fullscreen', self.full_screen)
    
    def close(self, event=None):
        self.running = False
        self.root.destroy()

# Initialize the app with full screen resolution and optional circle radius
if __name__ == "__main__":
    root = tk.Tk()

    app = CircleGridApp(root, circle_radius=60)
    root.mainloop()
