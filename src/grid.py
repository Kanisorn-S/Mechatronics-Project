import tkinter as tk

class GridApp:
    def __init__(self, root, grid_size=10):
        self.root = root
        self.root.title("Programmable Grid")

        # Default grid size, can be changed
        self.grid_size = grid_size

        # Window size and canvas
        self.window_width = 1280
        self.window_height = 720
        self.canvas = tk.Canvas(self.root, width=self.window_width, height=self.window_height, bg='white')
        self.canvas.pack()

        # Draw the grid
        self.draw_grid()

    def draw_grid(self):
        """Draws a grid with the current grid_size on the canvas."""
        self.canvas.delete("all")  # Clear any previous grid
        cell_width = self.window_width // self.grid_size
        cell_height = self.window_height // self.grid_size

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                # Draw rectangle for each grid cell
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")

    def update_grid(self, new_grid_size):
        """Update the grid size and redraw the grid."""
        self.grid_size = new_grid_size
        self.draw_grid()

# Create the application window
if __name__ == "__main__":
    root = tk.Tk()

    # Initialize the grid with default size 10x10
    app = GridApp(root, grid_size=50)

    # Example: Change the grid size to a different value programmatically
    # app.update_grid(20)  # Uncomment to change the grid to 20x20 programmatically

    root.mainloop()
