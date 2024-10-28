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

def calculate_XYZ(screen_width, screen_height, u,v):
                                      
  #Solve: From Image Pixels, find World Points

  uv_1=np.array([[u,v,1]], dtype=np.float32)
  uv_1=uv_1.T
  suv_1=scalingfactor*uv_1
  xyz_c=inverse_newcam_mtx.dot(suv_1)
  xyz_c=xyz_c-tvec1
  XYZ=inverse_R_mtx.dot(xyz_c)
  XYZ_t = convert_z(screen_width, screen_height, XYZ)

  return XYZ_t

def convert_z(screen_width, screen_height, xyz):
  x, y, z = xyz
  if int(z) == 492:
    return xyz
  print("x: ", x)
  print("y: ", y)
  print("z: ", z)
  a = np.abs((screen_width / 2) - x)
  b = np.abs((screen_height + 150) - y)
  c = np.sqrt(a**2 + b**2)
  print("a: ", a)
  print("b: ", b)
  r = (493 / z) * c
  print("r: ", r)
  theta = np.arctan(b / a)
  print("theta: ", theta)
  x_t = (screen_width / 2) - (r * np.cos(theta))
  if x > (screen_width / 2):
    x_t = (screen_width / 2) + (r * np.cos(theta))
  y_t = np.abs((screen_height) - (r * np.sin(theta)))
  xyz_t = [x_t, y_t, 493]
  return xyz_t

if __name__ == "__main__":
  
  cap  = cv.VideoCapture(1)

  print(calculate_XYZ(0, 0))

  while cap.isOpened():
    ret, frame = cap.read()
    cv.circle(frame, (400, 400), 2, (255, 0, 0), -1)
    print(calculate_XYZ(400, 400))
    cv.imshow("Webcam", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
  
  cap.release()
  cv.destroyAllWindows()