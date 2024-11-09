"""
This module contains utility functions for postprocessing nut detections.

Functions:
    adjust_coordinate(x, y, w, h, x_offset, y_offset, y_scale=1):
        Adjust the x, y coordinate based on the width, height, and offsets.
    
    convert_boxes(boxes):
        Convert an array [x, y, w, h] to an array [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    
    convert_contours(contours):
        Convert an array of contours, where each contour is an array of points (tuple), to an array of contour, where each contour is an array of points (np.array).
    
    calculate_contour_areas(points_arrays):
        Calculate the areas of contours given a list of points arrays.
    
    contour_to_points(contour):
        Convert an OpenCV contour to a list of points.
"""

import numpy as np
import cv2

def adjust_coordinate(x, y, w, h, x_offset, y_offset, y_scale=1):
    """
    Adjust the x, y coordinate based on the width, height, and offsets.
    
    Parameters:
        x (int): The x-coordinate.
        y (int): The y-coordinate.
        w (int): The width of the image.
        h (int): The height of the image.
        x_offset (int): The offset in the x-axis.
        y_offset (int): The offset in the y-axis.
    
    Returns:
        int: The new x-coordinate.
        int: The new y-coordinate.
    """
    mid_x = w / 2
    mid_y = h / 2
    x_diff = x - mid_x
    y_diff = y - mid_y
    new_x = x - x_offset * (y_diff / mid_y) * (x_diff / mid_x)
    new_y = y + y_offset - y_scale * abs(y_diff / mid_y)
    return int(new_x), int(new_y)

def convert_boxes(boxes):
    """
    Convert an array [x, y, w, h] to an array [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    
    Parameters:
        boxes (list of list of int): List of boxes, where each box is a list of integers [x, y, w, h].
    
    Returns:
        list of list of list of int: List of boxes, where each box is a list of points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    converted_boxes = []
    for box in boxes:
        x, y, w, h = box
        box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        converted_boxes.append(box)
    return converted_boxes
  
def convert_contours(contours):
    """
    Convert an array of contours, where each contour is an array of points (tuple), to an array of contour, where each contour is an array of points (np.array).
    
    Parameters:
        contours (list of list of tuple): List of contours, where each contour is a list of points (x, y).
    
    Returns:
        list of np.array: List of contours, where each contour is a numpy array of points (x, y).
    """
    converted_contours = []
    for contour in contours:
        converted_contour = np.array(contour)
        converted_contours.append(converted_contour)
    return converted_contours
  
  
def calculate_contour_areas(points_arrays):
    """
    Calculate the areas of contours given a list of points arrays.
    
    Parameters:
        points_arrays (list of np.array): List of numpy arrays, each containing points (x, y).
    
    Returns:
        np.array: Numpy array of contour areas.
    """
    areas = []
    for points in points_arrays:
        # Convert the points to a contour format expected by OpenCV
        contour = points.reshape((-1, 1, 2)).astype(np.int32)
        
        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        areas.append(area)
    
    # Convert the list of areas to a numpy array
    return np.array(areas)

def contour_to_points(contour):
    """
    Convert an OpenCV contour to a list of points.
    
    Parameters:
        contour (np.array): Numpy array containing points of the contour.
    
    Returns:
        list of tuple: List of points (x, y).
    """
    points = []
    for point in contour:
        x, y = point[0]
        points.append((x, y))
    return points
