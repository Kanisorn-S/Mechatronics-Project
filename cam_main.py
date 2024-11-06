import cv2 as cv
import numpy as np
from nut_classifier.model import find_nuts, calculate_contour_areas
from camconfirm import find_nut_circle
import joblib
import tkinter as tk

model = joblib.load('linear_regression_model.pkl')

crop_x, crop_y, crop_width, crop_height = 200, 300, 200, 100  # top-left x, y, width, height

cap = cv.VideoCapture(1)

while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break
  
  cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

  X = []
  boxes_X = []
  # circles = find_nut_circle(cropped_frame)
  # for circle in circles:
  #   x, y, _ = circle
  #   X.append([x, y])
  detected, blur, edged, boxes, centers, sizes = find_nuts(cropped_frame, max_size=200)
  for box in boxes:
    original_box = []
    for coor in box:
      original_box.append(coor + [crop_x, crop_y]) 
    boxes_X.append(original_box)
  np_centers = np.array(centers)
  shift = np.array([crop_x, crop_y])
  # for center in centers:
  #   original_center = np.add(np_centers, shift)
  #   X.append(original_center)
  center_X = np.add(np_centers, shift)
  # X = centers
  cv.imshow("Result", detected)
  cv.imshow("Blur", blur)
  cv.imshow("Edge", edged)

  Y = []
  boxes_Y = []
  boxes_X = np.array(boxes_X)
  for box in boxes_X:
    boxes_Y.append(model.predict(box))

  center_Y = model.predict(center_X)

  
  # (tl, tr, br, bl) = Y
  # size_ind = []
  # for size in sizes:
  #   if size < 50:
  #     size_ind.append(0) #M3
  #   elif 50 <= size < 92:
  #     size_ind.append(1) #M4
  #   elif size > 93:
  #     size_ind.append(2) #M5
  
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv.destroyAllWindows()

print(boxes_Y)
areas = calculate_contour_areas(boxes_Y)
print(areas)
size_ind = []
for area in areas:
  if area < 700:
    size_ind.append(0) #M3
  elif 700 <= area < 1050:
    size_ind.append(1) #M4
  elif 1050 <= area:
    size_ind.append(2) #M5
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
        for i, coor in enumerate(center_Y):
          self.draw_circle(coor, 30, size_ind[i])
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
        
    def draw_circle(self, center, radius, color_ind=0):
        """Helper function to draw a circle with a center dot"""
        colors = ['blue', 'red', 'green']
        sizes = ['M3', 'M4', 'M5']
        color = colors[color_ind]
        size = sizes[color_ind]
        x, y = center
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
        # Draw a dot at the center
        # self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color)
        # Display center coordinates as text
        self.canvas.create_text(x, y + radius + 10, text=f"{size}", fill=color)

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