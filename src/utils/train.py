"""
This module contains utility functions used to train models 

Functions:
    add_to_csv(center_XX, center_XY, center_YX, center_YY, bounding_box_size_X, bounding_box_size_Y, min_box_size_X, min_box_size_Y, contour_size_X, contour_size_Y, nut_type):
        Adds the provided data to a CSV file named 'nuts.csv'.
"""

def add_to_csv(center_XX, center_XY, center_YX, center_YY, bounding_box_size_X, bounding_box_size_Y, min_box_size_X, min_box_size_Y, contour_size_X, contour_size_Y, nut_type):
    """
    Add the data to a CSV file.
    
    Parameters:
        center_XX (int): The x-coordinate of the center in the camera frame.
        center_XY (int): The y-coordinate of the center in the camera frame.
        center_YX (int): The x-coordinate of the center in the projected frame.
        center_YY (int): The y-coordinate of the center in the projected frame.
        bounding_box_size_X (int): The size of the bounding box in the x-axis.
        bounding_box_size_Y (int): The size of the bounding box in the y-axis.
        min_box_size_X (int): The size of the minimum box in the x-axis.
        min_box_size_Y (int): The size of the minimum box in the y-axis.
        contour_size_X (int): The size of the contour in the x-axis.
        contour_size_Y (int): The size of the contour in the y-axis.
        nut_type (str): The type of nut.
    """
    with open("nuts.csv", "a") as file:
        file.write(f"{center_XX}, {center_XY}, {center_YX}, {center_YY}, {bounding_box_size_X}, {bounding_box_size_Y}, {min_box_size_X}, {min_box_size_Y}, {contour_size_X}, {contour_size_Y}, {nut_type}\n")
    
    file.close()