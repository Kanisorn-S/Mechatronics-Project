"""
This module contains utility functions for detecting nuts in an image using computer vision techniques.

Functions:
- find_nuts(image, min_size=0, max_size=10000000000): Detects nuts in the given image and returns various details about the detected nuts.
- show_images(images): Displays a list of images in separate windows.
"""
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from utils.postprocess import contour_to_points

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