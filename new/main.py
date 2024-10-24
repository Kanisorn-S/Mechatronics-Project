import os
import cv2 as cv
import numpy as np


#load camera calibration
savedir = "camera_info"
cam_mtx=np.load(os.path.join(savedir, 'cam_mtx.npy'))
dist=np.load(os.path.join(savedir, 'dist.npy'))
newcam_mtx=np.load(os.path.join(savedir, 'newcam_mtx.npy'))
roi=np.load(os.path.join(savedir, 'roi.npy'))
rvec1=np.load(os.path.join(savedir, 'rvec1.npy'))
tvec1=np.load(os.path.join(savedir, 'tvec1.npy'))
R_mtx=np.load(os.path.join(savedir, 'R_mtx.npy'))
Rt=np.load(os.path.join(savedir, 'Rt.npy'))
P_mtx=np.load(os.path.join(savedir, 'P_mtx.npy'))

s_arr=np.load(os.path.join(savedir, 's_arr.npy'))
scalingfactor=s_arr[0]

inverse_newcam_mtx = np.linalg.inv(newcam_mtx)
inverse_R_mtx = np.linalg.inv(R_mtx)

def calculate_XYZ(u,v):
                                      
  #Solve: From Image Pixels, find World Points

  uv_1=np.array([[u,v,1]], dtype=np.float32)
  uv_1=uv_1.T
  suv_1=scalingfactor*uv_1
  xyz_c=inverse_newcam_mtx.dot(suv_1)
  xyz_c=xyz_c-tvec1
  XYZ=inverse_R_mtx.dot(xyz_c)

  return XYZ

cap  = cv.VideoCapture(0)

print(calculate_XYZ(0, 0))

while cap.isOpened():
  ret, frame = cap.read()
  cv.imshow("Webcam", frame)
  
  if cv.waitKey(1) & 0xFF == ord('q'):
    break
  
cap.release()
cv.destroyAllWindows()