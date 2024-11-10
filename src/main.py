import cv2 as cv
import numpy as np
import torch
from torch import nn
import joblib
import tkinter as tk
from collections import Counter
import os
from utils.postprocess import convert_contours, calculate_contour_areas, convert_boxes, adjust_coordinate
from utils.model import find_nuts
from utils.train import add_to_csv

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(current_dir, './model/models/new_linear_regression_model.pkl'))

# Nut types to class
# M3 = 0
# M4 = 1
# M5 = 2
training_mode = False
trained_class = 2

NEURAL_NETWORK = 'neural_network'
NAIVE_BAYES = 'naive_bayes'
DECISION_TREE = 'decision_tree'
RANDOM_FOREST = 'random_forest'
SVM = 'svm'
KNN = 'knn'

# Choose model to use to predict
classification_model = {
  'neural_network': './model/models/nut_classifier_weights.pth',
  'naive_bayes': './model/models/naive_bayes_model.pkl',
  'decision_tree': './model/models/decision_tree_model.pkl',
  'random_forest': './model/models/random_forest_model.pkl',
  'svm': './model/models/svm_model.pkl',
  'knn': './model/models/knn_model.pkl'
}
# Choose the model to use
model_choice = SVM  # Change to 'neural_network', 'naive_bayes', 'decision_tree', 'random_forest', 'svm', or 'knn'
model_path = classification_model[model_choice]


# Define the crop regions - adjust as needed
crop_regions = {      # top-left x, y, width, height
    "none": (0, 0, 640, 480),  
    "top": (200, 300, 200, 100),  
    "bottom": (250, 150, 200, 100),  
    "all": (100, 150, 400, 250),
    "center": (200, 200, 200, 200)
}

# Choose the crop region
crop_choice = "all"  # Change to "none", "top", "bottom", "all", or "center"
crop_region = crop_regions[crop_choice]

cap = cv.VideoCapture(0)

while cap.isOpened():
  
  # Read frame from camera
  ret, frame = cap.read()
  if not ret:
    break
  
  # Crop the frame to the defined region if cropping is enabled
  crop_x, crop_y, crop_width, crop_height = crop_region
  cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]

  # Find nuts in the cropped frame
  detected, blur, edged, min_boxes, centers, min_box_sizes, contours, contour_sizes, bounding_boxes, bounding_box_sizes = find_nuts(cropped_frame, max_size=200)
  
  # Process the contours
  # Transform contours coordinates
  contours = convert_contours(contours)
  original_contours = []
  for contour in contours:
    original_contour = []
    for coor in contour:
      original_contour.append(coor + [crop_x, crop_y])
    original_contours.append(original_contour)

  contours_Y = []
  contours_X = original_contours
  for contour in contours_X:
    contours_Y.append(model.predict(contour))

  # Contour sizes
  np_contour_sizes = np.array(contour_sizes)
  contour_sizes_X = np_contour_sizes

  # Process the mininmum boxes
  min_boxes_X = []
  for box in min_boxes:
    original_box = []
    for coor in box:
      original_box.append(coor + [crop_x, crop_y]) 
    min_boxes_X.append(original_box)
  min_boxes_Y = []
  min_boxes_X = np.array(min_boxes_X)
  for box in min_boxes_X:
    min_boxes_Y.append(model.predict(box))
  
  # Min box sizes
  np_min_box_sizes = np.array(min_box_sizes)
  min_box_sizes_X = np_min_box_sizes

  # Process the bounding boxes
  original_box_coords = []
  for box in bounding_boxes:
    original_box = np.add(box, [crop_x, crop_y, 0, 0])
    original_box_coords.append(original_box)
  bounding_boxes_X = convert_boxes(original_box_coords)
    # original_box_coords = convert_boxes([original_box])
    # bounding_boxes_X.append(original_box_coords)
  bounding_boxes_Y = []
  bounding_boxes_X = np.array(bounding_boxes_X)
  for box in bounding_boxes_X:
    bounding_boxes_Y.append(model.predict(box))
  
  # Bounding box sizes
  np_bounding_box_sizes = np.array(bounding_box_sizes)
  bounding_box_sizes_X = np_bounding_box_sizes


  # Process the centers
  np_centers = np.array(centers)
  shift = np.array([crop_x, crop_y])
  center_X = np.add(np_centers, shift)
  center_Y = model.predict(center_X)

  # Display the detection results
  cv.imshow("Result", detected)
  cv.imshow("Blur", blur)
  cv.imshow("Edge", edged)
  
  if cv.waitKey(1) & 0xFF == ord('q'):
    break

# Clean up openCV
cap.release()
cv.destroyAllWindows()

# List of variables
# center_Y => Coordinates of the center of the detected nuts in projector frame
# bounding_box_sizes_X => Sizes of the bounding boxes of the detected nuts in camera frame
# bounding_boxes_Y => Coordinates of the bounding boxes of the detected nuts in projector frame
# min_box_sizes_X => Sizes of the minimum boxes of the detected nuts in camera frame
# min_boxes_Y => Coordinates of the minimum boxes of the detected nuts in projector frame
# contour_sizes_X => Sizes of the contours of the detected nuts in camera frame
# contours_Y => Coordinates of the contours of the detected nuts in projector frame

# Process the contours
contour_sizes_Y = calculate_contour_areas(contours_Y)

# Process the mininmum boxes
min_box_sizes_Y = calculate_contour_areas(min_boxes_Y)

# Process the bounding boxes
bounding_box_sizes_Y = calculate_contour_areas(bounding_boxes_Y)

# Print out the results
print("Center of the detected nuts in camera frame:")
print(center_X)
print("Center of the detected nuts in projector frame:")
print(center_Y)
print("Sizes of the bounding boxes of the detected nuts in camera frame:")
print(bounding_box_sizes_X)
print("Sizes of the bounding boxes of the detected nuts in projector frame:")
print(bounding_box_sizes_Y)
print("Sizes of the minimum boxes of the detected nuts in camera frame:")
print(min_box_sizes_X)
print("Sizes of the minimum boxes of the detected nuts in projector frame:")
print(min_box_sizes_Y)
print("Sizes of the contours of the detected nuts in camera frame:")
print(contour_sizes_X)
print("Sizes of the contours of the detected nuts in projector frame:")
print(contour_sizes_Y)

# Split the array center_X into two arrays center_XX and center_XY, where center_XX contains the X coordinates and center_XY contains the Y coordinates, keeping the order of the nuts
center_XX = center_X[:, 0]
center_XY = center_X[:, 1]

# Split the array center_Y into two arrays center_YX and center_YY, where center_YX contains the X coordinates and center_YY contains the Y coordinates, keeping the order of the nuts
center_YX = center_Y[:, 0]
center_YY = center_Y[:, 1]

# transform the list of bounding box sizes, min box sizes, contour sizes to a numpy array of 
# nuts where each nut is an array [bounding_box_size_X, bounding_box_size_Y, min_box_size_X, min_box_size_Y, contour_size_X, contour_size_Y]
nuts = np.array([center_XX, center_XY, center_YX, center_YY, bounding_box_sizes_X, bounding_box_sizes_Y, min_box_sizes_X, min_box_sizes_Y, contour_sizes_X, contour_sizes_Y]).T

# Use neural network to predict the type of nut
# Load the scaler and the model
scaler = joblib.load('./model/models/scaler.pkl')
# model = joblib.load(model_path)

# Standardize the nuts
standardized_nuts = scaler.transform(nuts)

# Try all models and vote for the most common prediction
models = [NAIVE_BAYES, DECISION_TREE, RANDOM_FOREST, SVM, KNN]
predictions = []
for model_name in models:
  model_path = classification_model[model_name]
  model = joblib.load(model_path)
  predicted_classes = model.predict(standardized_nuts)
  predictions.append(predicted_classes)

# Load the neural network model
class NutClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NutClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model with the correct input size and number of classes
input_size = 10
num_classes = 3  
model = NutClassifier(input_size=input_size, num_classes=num_classes)

# Load the model weights
model.load_state_dict(torch.load('./model/models/nut_classifier_weights.pth'))
model.eval()

# Convert the standardized nuts to a tensor
standardized_nuts = torch.tensor(standardized_nuts, dtype=torch.float32)

# Predict the type of nut
nut_type = {
    0: 'M3',
    1: 'M4',
    2: 'M5'
}
outputs = model(standardized_nuts)
_, predicted = torch.max(outputs, 1)

print("--------------- Neural Network -------------------")
print(predicted.tolist())
predicted_classes = [nut_type[pred] for pred in predicted.tolist()]
print(f'The predicted nut type is: {predicted_classes}')
print("--------------------------------------------------")

# create a variable for stroing nut type
nut_types = predicted.tolist()
predictions.append(nut_types)

# Transform the predictions
predictions = np.array(predictions).T

# Most common prediction
most_common_prediction = []

for prediction in predictions:
  counter = Counter(prediction)
  most_common_prediction.append(counter.most_common(1)[0][0])

# Predict the class
# predicted_classes = model.predict(standardized_nuts)
print(f'The predicted nut type is: {most_common_prediction}')



if training_mode:
  #  Add more nuts to the csv file
  for i in range(len(bounding_box_sizes_X)):
    add_to_csv(center_XX[i], center_XY[i], center_YX[i], center_YY[i], bounding_box_sizes_X[i], bounding_box_sizes_Y[i], min_box_sizes_X[i], min_box_sizes_Y[i], contour_sizes_X[i], contour_sizes_Y[i], trained_class)


# Tkinter for display
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
          self.draw_circle(coor, 30, most_common_prediction[i])
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
        x_offset = 125
        y_offset = 20
        y_scale = 25
        colors = ['blue', 'red', 'green']
        sizes = ['M3', 'M4', 'M5']
        color = colors[color_ind]
        size = sizes[color_ind]
        x, y = center
        x, y = adjust_coordinate(x, y, self.screen_width, self.screen_height, x_offset, y_offset, y_scale)
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)
        # Draw a dot at the center
        # self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color)
        # Display center coordinates as text
        self.canvas.create_text(x, y + radius + 10, text=f"{size}", fill=color)

    # write a method that draws a line across the center of the screen
    def draw_center_lines(self):
        """Draw a line across the center of the screen"""
        self.canvas.create_line(0, self.screen_height // 2, self.screen_width, self.screen_height // 2, fill='black', width=2)
        
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