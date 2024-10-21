import cv2 as cv
from object_detector import *
import numpy as np

# Load Aruco detector
parameters = cv.aruco.DetectorParameters()
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)


# Load Object Detector
detector = HomogeneousBgDetector()

cap = cv.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    
    corners, _, _ = cv.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    
    int_corners = np.intp(corners)
    cv.polylines(frame, int_corners, True, (0, 255, 0), 5)
    
    # aruco_perimeter = cv.arcLength(corners[0], True)
    
    # pixel_cm_ratio = aruco_perimeter / (9.6 * 4)
    
    # contours = detector.detect_objects(frame)
    
    # for cnt in contours:
    #     rect = cv.minAreaRect(cnt)
    #     (x, y), (w, h), angle = rect
        
    #     object_width = w / pixel_cm_ratio
    #     object_height = h / pixel_cm_ratio
        
    #     box = cv.boxPoints(rect)
    #     box = np.intp(box)
        
    #     cv.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    #     cv.polylines(frame, [box], True, (255, 0, 0), 2)
    #     cv.putText(frame, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    #     cv.putText(frame, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

    cv.imshow("Webcam", frame)
        
     
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()



