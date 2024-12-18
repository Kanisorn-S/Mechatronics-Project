import tkinter as tk

class CircleGridApp:
    def __init__(self, root, circle_radius=50):
        self.root = root
        self.root.title("Circle Grid")

        # Get the screen resolution
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        print(self.screen_width)
        print(self.screen_height)

        # Set the initial resolution to full screen size
        self.root.geometry(f"{self.screen_width}x{self.screen_height}")

        # Canvas to draw the grid and circles
        self.canvas = tk.Canvas(root, width=self.screen_width, height=self.screen_height, bg='white')
        self.canvas.pack()

        # Circle parameters
        self.circle_radius = circle_radius
        self.grid_size = 6
        self.spacing = min(self.screen_width, self.screen_height) // (self.grid_size + 1)
        self.centers = []  # To store programmable centers of circles

        # Boolean to toggle the 10th circle manually in code
        self.show_10th_circle = False
        # self.tenth_circle_center = (666, 473)  # Programmable center
        self.tenth_circle_center = (1018, 810)  # Programmable center

        # Draw the initial 9 circles
        # self.draw_cross()
        self.draw_circles()
        # self.draw_line()
        # self.draw_poly((453, 533), (487, 526), (497, 560), (463, 571))
        # self.draw_poly((595, 728), (619, 728), (619, 754), (595, 754))
        # self.draw_poly((600, 519), (630, 519), (630, 545), (600, 545))
        # self.draw_poly((684, 646), (708, 646), (708, 672), (684, 672))
        # self.draw_poly((749, 523), (780, 497), (800, 523), (770, 549))
        # self.draw_poly((780, 643), (810, 632), (820, 665), (789, 676))

        # Bind 'p' key for full screen toggle
        self.root.bind('<p>', self.toggle_full_screen)
        self.root.bind('<Escape>', self.close)

        # Track full screen state
        self.full_screen = False

    def draw_line(self):
        x_position = 100
        self.canvas.create_line(321, 467, 1002, 468, fill="black", width=2)
    
    def draw_poly(self, point_A, point_B, point_C, point_D):
        # point_A = (457, 541)    # Top left corner
        # point_B = (481, 526)   # Top right corner
        # point_C = (494, 560)  # Bottom right corner
        # point_D = (470, 571)   # Bottom left corner

        # Draw the rectangle using points A, B, C, and D
        self.canvas.create_polygon(
            point_A[0], point_A[1],
            point_B[0], point_B[1],
            point_C[0], point_C[1],
            point_D[0], point_D[1],
            outline="blue", fill="", width=2
        )
        
    def draw_circle(self, center, radius, color='black'):
        """Helper function to draw a circle with a center dot"""
        x, y = center
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color)
        # Draw a dot at the center
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color)
        # Display center coordinates as text
        self.canvas.create_text(x, y + radius + 10, text=f"({x}, {y})", fill=color)

    def draw_grid_lines(self):
        """Draw lines through the centers of all circles, creating a grid"""
        for center in self.centers:
            x, y = center
            # Draw vertical line through the center
            self.canvas.create_line(x, 0, x, self.screen_height, fill="gray", dash=(4, 2))
            # Draw horizontal line through the center
            self.canvas.create_line(0, y, self.screen_width, y, fill="gray", dash=(4, 2))

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

        # Draw grid lines through the centers of all circles
        self.draw_grid_lines()

    def draw_programmable_circle(self, x, y, radius=None, color='blue'):
        """Draw a circle at programmable coordinates (x, y)"""
        if radius is None:
            radius = self.circle_radius
        self.draw_circle((x, y), radius, color)

    def toggle_full_screen(self, event):
        """Toggle between full screen and windowed mode on 'p' key press"""
        self.full_screen = not self.full_screen
        self.root.attributes('-fullscreen', self.full_screen)
    
    def close(self, event=None):
        self.running = False
        self.root.destroy()

    # write a function to draw 2 lines, one vertical and one horizontal, that intersect at the center of the screen
    def draw_cross(self):
        # Draw the vertical line
        self.canvas.create_line(self.screen_width // 2, 0, self.screen_width // 2, self.screen_height, fill="black", width=2)
        # Draw the horizontal line
        self.canvas.create_line(0, self.screen_height // 2, self.screen_width, self.screen_height // 2, fill="black", width=2)

# Initialize the app with full screen resolution and optional circle radius
if __name__ == "__main__":
    root = tk.Tk()

    app = CircleGridApp(root, circle_radius=60)
    app.draw_programmable_circle(237, 165)  # Example usage
    root.mainloop()