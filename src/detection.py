import cv2 as cv
import numpy as np
import torch
from torch import nn
import joblib
import tkinter as tk
from collections import Counter
from utils.postprocess import convert_contours, calculate_contour_areas, convert_boxes, adjust_coordinate
from utils.model import find_nuts
from utils.train import add_to_csv
import os

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
  'neural_network': os.path.join(current_dir, './model/models/nut_classifier_weights.pth'),
  'naive_bayes': os.path.join(current_dir, './model/models/naive_bayes_model.pkl'),
  'decision_tree': os.path.join(current_dir, './model/models/decision_tree_model.pkl'),
  'random_forest': os.path.join(current_dir, './model/models/random_forest_model.pkl'),
  'svm': os.path.join(current_dir, './model/models/svm_model.pkl'),
  'knn': os.path.join(current_dir, './model/models/knn_model.pkl')
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
    "center": (200, 200, 200, 200),
    "real": (200, 150, 325, 250),
}

# Choose the crop region
crop_choice = "all"  # Change to "none", "top", "bottom", "all", or "center"
crop_region = crop_regions[crop_choice]

def detect_nuts(frame, crop_region):
    crop_x, crop_y, crop_width, crop_height = crop_region
    cropped_frame = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
    return find_nuts(cropped_frame, max_size=200)

def process_nuts(detected, blur, edged, min_boxes, centers, min_box_sizes, contours, contour_sizes, bounding_boxes, bounding_box_sizes, crop_region):
    crop_x, crop_y, crop_width, crop_height = crop_region

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

    np_contour_sizes = np.array(contour_sizes)
    contour_sizes_X = np_contour_sizes

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

    np_min_box_sizes = np.array(min_box_sizes)
    min_box_sizes_X = np_min_box_sizes

    original_box_coords = []
    for box in bounding_boxes:
        original_box = np.add(box, [crop_x, crop_y, 0, 0])
        original_box_coords.append(original_box)
    bounding_boxes_X = convert_boxes(original_box_coords)
    bounding_boxes_Y = []
    bounding_boxes_X = np.array(bounding_boxes_X)
    for box in bounding_boxes_X:
        bounding_boxes_Y.append(model.predict(box))

    np_bounding_box_sizes = np.array(bounding_box_sizes)
    bounding_box_sizes_X = np_bounding_box_sizes

    np_centers = np.array(centers)
    shift = np.array([crop_x, crop_y])
    center_X = np.add(np_centers, shift)
    center_Y = model.predict(center_X)

    contour_sizes_Y = calculate_contour_areas(contours_Y)
    min_box_sizes_Y = calculate_contour_areas(min_boxes_Y)
    bounding_box_sizes_Y = calculate_contour_areas(bounding_boxes_Y)

    center_XX = center_X[:, 0]
    center_XY = center_X[:, 1]
    center_YX = center_Y[:, 0]
    center_YY = center_Y[:, 1]

    nuts = np.array([center_XX, center_XY, center_YX, center_YY, bounding_box_sizes_X, bounding_box_sizes_Y, min_box_sizes_X, min_box_sizes_Y, contour_sizes_X, contour_sizes_Y]).T
    return nuts, center_Y 

def predict_nut_types(nuts):
    scaler = joblib.load(os.path.join(current_dir, './model/models/scaler.pkl'))
    standardized_nuts = scaler.transform(nuts)

    models = [NAIVE_BAYES, DECISION_TREE, RANDOM_FOREST, SVM, KNN]
    predictions = []
    for model_name in models:
        model_path = classification_model[model_name]
        model = joblib.load(model_path)
        predicted_classes = model.predict(standardized_nuts)
        predictions.append(predicted_classes)

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

    input_size = 10
    num_classes = 3  
    model = NutClassifier(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(current_dir, './model/models/nut_classifier_weights.pth')))
    model.eval()

    standardized_nuts = torch.tensor(standardized_nuts, dtype=torch.float32)
    outputs = model(standardized_nuts)
    _, predicted = torch.max(outputs, 1)

    nut_type = {
        0: 'M3',
        1: 'M4',
        2: 'M5'
    }
    predicted_classes = [nut_type[pred] for pred in predicted.tolist()]
    predictions.append(predicted.tolist())

    predictions = np.array(predictions).T
    most_common_prediction = []
    for prediction in predictions:
        counter = Counter(prediction)
        most_common_prediction.append(counter.most_common(1)[0][0])

    return most_common_prediction

# Example usage
if __name__ == "__main__":
    cap = cv.VideoCapture(1)
    crop_region = crop_regions["all"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected, blur, edged, min_boxes, centers, min_box_sizes, contours, contour_sizes, bounding_boxes, bounding_box_sizes = detect_nuts(frame, crop_region)
        nuts = process_nuts(detected, blur, edged, min_boxes, centers, min_box_sizes, contours, contour_sizes, bounding_boxes, bounding_box_sizes, crop_region)
        predictions = predict_nut_types(nuts)

        cv.imshow("Result", detected)
        cv.imshow("Blur", blur)
        cv.imshow("Edge", edged)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
