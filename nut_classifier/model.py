from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# write a function that takes 7 inputs, bounding_box_size_X, bounding_box_size_Y, min_box_size_X, min_box_size_Y, contour_size_X, contour_size_Y, and nut_type
# The function should add the data into a csv file called "nuts.csv"
def add_to_csv(bounding_box_size_X, bounding_box_size_Y, min_box_size_X, min_box_size_Y, contour_size_X, contour_size_Y, nut_type):
    """
    Add the data to a CSV file.
    
    Parameters:
        bounding_box_size_X (int): The size of the bounding box in the x-axis.
        bounding_box_size_Y (int): The size of the bounding box in the y-axis.
        min_box_size_X (int): The size of the minimum box in the x-axis.
        min_box_size_Y (int): The size of the minimum box in the y-axis.
        contour_size_X (int): The size of the contour in the x-axis.
        contour_size_Y (int): The size of the contour in the y-axis.
        nut_type (str): The type of nut.
    """
    with open("nuts.csv", "a") as file:
        file.write(f"{bounding_box_size_X}, {bounding_box_size_Y}, {min_box_size_X}, {min_box_size_Y}, {contour_size_X}, {contour_size_Y}, {nut_type}\n")
    
    file.close()

# write me a function that takes a coordinate (x, y), the width and height of an image (w, h), and two offsets (x_offset, y_offset)
# The function should adjust (x, y) so that if (x, y) is at the center, it is offset by (x_offset, y_offset)
# The further away from the center in the y-axis, the smaller the offset in the y-axis linearly
# For x, if (x, y) is above the center, the x-coordinate gets offset away from the center
# For x, if (x, y) is below the center, the x-coordinate gets offset towards the center
# The further away from the center in the y-axis, the larger the offset in the x-axis linearly
# The function should return the new (x, y) coordinate
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

# write me a function to convert an array [x, y, w, h] to an array [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
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
  
# write me a function to convert an array of contours, where each contour is an array of points (tuple), to an array of contour, where each contour is an array of points (np.array)
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

# Write a function to convert opencv contour to a list of points
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

def find_nuts(image, min_size=0, max_size=10000000000):
  
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray, (5, 5), 0)

  edged = cv2.Canny(blur, 50, 100)
  edged = cv2.dilate(edged, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)

  # show_images([blur, edged])

  # Find contours
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  # Sort contours from left to right as leftmost contour is reference object
  # (cnts, _) = contours.sort_contours(cnts)

  # Remove contours which are not large enough
  # cnts = [x for x in cnts if cv2.contourArea(x) < max_size and cv2.contourArea(x) > min_size]
  # for cnt in cnts:
  #   print(cv2.contourArea(cnt))

  min_boxes = []
  centers = []
  min_box_sizes = []
  contour_sizes = []
  contours = []
  bounding_boxes = []
  bounding_boxes_size = []
  for cnt in cnts:
    # bounding boxes
    (x, y, w, h) = cv2.boundingRect(cnt)
    bounding_boxes.append([x, y, w, h])
    bounding_boxes_size.append(w * h)
    contour_sizes.append(cv2.contourArea(cnt))
    contour = contour_to_points(cnt)
    contours.append(contour)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    box = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    min_boxes.append(box)
    cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
    mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
    mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
    wid = euclidean(tl, tr)
    ht = euclidean(tr, br)
    mid_x = x + (w / 2)
    mid_y = y + (h / 2)
    centers.append([mid_x, mid_y])
    box_size = wid * ht
    min_box_sizes.append(box_size)
    # print(mid_pt_horizontal)
    # print(mid_pt_verticle)
    cv2.putText(image, "{:.1f}px".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(image, "{:.1f}px".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

  return image, blur, edged, min_boxes, centers, min_box_sizes, contours, contour_sizes, bounding_boxes, bounding_boxes_size

# Function to show array of images (intermediate results)
def show_images(images):
	for i, img in enumerate(images):
		cv2.imshow("image_" + str(i), img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
  img_path = "./image.png"

  # Read image and preprocess
  # image = cv2.imread(img_path)
  cap = cv2.VideoCapture(1)

  while cap.isOpened():
    
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # show_images([blur, edged])

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)

    # Remove contours which are not large enough
    # cnts = [x for x in cnts if cv2.contourArea(x) > 500]

    cv2.drawContours(image, cnts, -1, (0,255,0), 3)

    # show_images([image, edged])
    # print(len(cnts))

    # min_area = 10000
    # for cnt in cnts:
    #   (x, y, w, h) = cv2.boundingRect(cnt)
    #   area = w * h
    #   print(area)
    #   if area > min_area: 
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # show_images([image, edged])

    # Reference object dimensions
    # Here for reference I have used a 2cm x 2cm square
    # ref_object = cnts[0]
    # box = cv2.minAreaRect(ref_object)
    # box = cv2.boxPoints(box)
    # box = np.array(box, dtype="int")
    # box = perspective.order_points(box)
    # (tl, tr, br, bl) = box
    # dist_in_pixel = euclidean(tl, tr)
    # dist_in_cm = 2
    # pixel_per_cm = dist_in_pixel/dist_in_cm

    # min_area = 500
    # cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > min_area]
    # Draw remaining contours
    for cnt in cnts:
      # (x, y, w, h) = cv2.boundingRect(cnt)
      # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
      box = cv2.minAreaRect(cnt)
      box = cv2.boxPoints(box)
      box = np.array(box, dtype="int")
      box = perspective.order_points(box)
      (tl, tr, br, bl) = box
      cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
      mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
      mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
      wid = euclidean(tl, tr)
      ht = euclidean(tr, br)
      print(mid_pt_horizontal)
      print(mid_pt_verticle)
      cv2.putText(image, "{:.1f}px".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
      cv2.putText(image, "{:.1f}px".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # show_images([image])
    cv2.imshow("Image", image)
    cv2.imshow("Edge", edged)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  cap.release()
  cv2.destroyAllWindows()