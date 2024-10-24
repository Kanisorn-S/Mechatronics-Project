import os
import numpy as np 
import cv2 as cv 
import cv2.aruco as aruco 
import math 

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

# ArUco marker size
marker_size = 97 

# with open('camera_cal.npy', 'rb') as f:
#     camera_matrix = np.load(f)
#     camera_distortion = np.load(f)

savedir="camera_info"
camera_matrix=np.load(os.path.join(savedir,'cam_mtx.npy'))
camera_distortion=np.load(os.path.join(savedir, 'dist.npy'))

# ArUco markers dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

cap = cv.VideoCapture(0)

# Camera Specs
# camera_width = 640
# camera_height = 480
# camera_frame_rate = 40

# cap.set(2, camera_width)
# cap.set(4, camera_height)
# cap.set(5, camera_frame_rate)

printed = False
while cap.isOpened():

    ret, frame = cap.read()

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(gray_frame, aruco_dict, camera_matrix, camera_distortion)

    if ids is not None:

        aruco.drawDetectedMarkers(frame, corners)
        cv.circle(frame, (int(corners[0][0][0][0]), int(corners[0][0][0][1])), 5, (255, 0, 0), -1)

        rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        if not printed:
            print(rvec_list_all[0])
            print(tvec_list_all[0])
            printed = True

        rvec = rvec_list_all[0][0]
        tvec = tvec_list_all[0][0]

        # aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 100)
        # cv.drawFrameAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 0.1)
        # rvec_flipped = rvec * -1
        # tvec_flipped = tvec * -1
        rotation_matrix, jacobian = cv.Rodrigues(rvec)
        realworld_tvec = np.dot(rotation_matrix, tvec)

        pitch, roll, yaw = rotationMatrixToEulerAngles(rotation_matrix)

        # tvec_str = "x=%4.0f, y=%4.0f, direction=%4.0f"%(realworld_tvec[0], realworld_tvec[1], math.degrees(yaw))
        tvec_str = "x=%4.0f, y=%4.0f, z=%4.0f"%(tvec[0], tvec[1], tvec[2])
        cv.putText(frame, tvec_str, (20, 400), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
    
    cv.imshow("frame", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv.destroyAllWindows()