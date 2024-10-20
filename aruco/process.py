import numpy as np 
import cv2 as cv 
import glob 

# Chess board information
cb_width = 9
cb_height = 6
cb_square_size = 26.3

# Termination Criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
cb_3d_points = np.zeros((cb_width * cb_height, 3), np.float32)
cb_3d_points[:, :2] = np.mgrid[0:cb_width, 0:cb_height].T.reshape(-1, 2) * cb_square_size

# Arrays to stor object points and image points
list_cb_3d_points = []
list_cb_2d_img_points = []

list_images = glob.glob('images/*.jpg')

for frame_name in list_images:
    img = cv.imread(frame_name)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points
    if ret == True:

        list_cb_3d_points.append(cb_3d_points)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        list_cb_2d_img_points.append(corners2)

        # Draw and display
        cv.drawChessboardCorners(img, (cb_width, cb_height), corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(list_cb_3d_points, list_cb_2d_img_points, gray.shape[::-1], None, None)

print("Calibration Matrix: ")
print(mtx)
print("Distortion: ", dist)

with open('camera_cal.npy', 'wb') as f:
    np.save(f, mtx)
    np.save(f, dist)