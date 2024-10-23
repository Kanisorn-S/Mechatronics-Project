import numpy as np 
import cv2 as cv 
import glob 
import os

OUT_IMAGE_DIR = "new_out_images"

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

list_images = glob.glob('calibration_images/*.jpg')

for i, frame_name in enumerate(list_images):
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
        file_name = "out_image_" + str(i) + ".jpg"
        path = os.path.join(OUT_IMAGE_DIR, file_name)
        cv.imwrite(path, img)
        cv.imshow("img", img)
        cv.waitKey(500)
    
    img1=img

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(list_cb_3d_points, list_cb_2d_img_points, gray.shape[::-1], None, None)

savedir = "camera_info"

print("Calibration Matrix: ")
print(mtx)
np.save(os.path.join(savedir, 'cam_mtx.npy'), mtx)

print("Distortion: ", dist)
np.save(os.path.join(savedir, 'dist.npy'), dist)

print("r vecs")
print(rvecs[2])

print("t Vecs")
print(tvecs[2])



print(">==> Calibration ended")


h,  w = img1.shape[:2]
print("Image Width, Height")
print(w, h)
#if using Alpha 0, so we discard the black pixels from the distortion.  this helps make the entire region of interest is the full dimensions of the image (after undistort)
#if using Alpha 1, we retain the black pixels, and obtain the region of interest as the valid pixels for the matrix.
#i will use Apha 1, so that I don't have to run undistort.. and can just calculate my real world x,y
newcam_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

print("Region of Interest")
print(roi)
np.save(os.path.join(savedir, 'roi.npy'), roi)

print("New Camera Matrix")
#print(newcam_mtx)
np.save(os.path.join(savedir, 'newcam_mtx.npy'), newcam_mtx)
print(np.load(os.path.join(savedir, 'newcam_mtx.npy')))

inverse = np.linalg.inv(newcam_mtx)
print("Inverse New Camera Matrix")
print(inverse)


# undistort
undst = cv.undistort(img1, mtx, dist, None, newcam_mtx)

# crop the image
x, y, w, h = roi
dist = dist[y:y+h, x:x+w]
cv.circle(dist,(308,160),5,(0,255,0),2)
cv.imshow('img1', img1)
cv.waitKey(5000)      
cv.destroyAllWindows()
cv.imshow('img1', undst)
cv.waitKey(5000)      
cv.destroyAllWindows()