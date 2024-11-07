import numpy as np
from sklearn.linear_model import LinearRegression
import joblib  # for saving the model
import os

# #load camera calibration
# savedir="camera_info"
# cam_mtx=np.load(os.path.join(savedir,'cam_mtx.npy'))
# dist=np.load(os.path.join(savedir, 'dist.npy'))
# newcam_mtx=np.load(os.path.join(savedir, 'newcam_mtx.npy'))
# roi=np.load(os.path.join(savedir, 'roi.npy'))


# #load center points from New Camera matrix
# cx=newcam_mtx[0,2]
# cy=newcam_mtx[1,2]
# fx=newcam_mtx[0,0]

# print("cx: "+str(cx)+",cy "+str(cy)+",fx "+str(fx))
# Sample data (replace this with your actual data)
X_center=437
Y_center=515
Z_center=486
world_points = [
  [400, 130], [528, 130], [656, 130], [784, 130], [912, 130], [1040, 130],
  [400, 258], [528, 258], [656, 258], [784, 258], [912, 258], [1040, 258],
  [400, 386], [528, 386], [656, 386], [784, 386], [912, 386], [1040, 386],
  [400, 514], [528, 514], [656, 514], [784, 514], [912, 514], [1040, 514],
  [400, 642], [528, 642], [656, 642], [784, 642], [912, 642], [1040, 642],
  [400, 770], [528, 770], [656, 770], [784, 770], [912, 770], [1040, 770],
]

cam_points = [
  [415, 380], [370, 380], [325, 380], [280, 382], [236, 383], [190, 385],
  [420, 350], [373, 350], [325, 350], [280, 350], [232, 350], [185, 350],
  [423, 313], [375, 314], [327, 314], [278, 313], [228, 314], [177, 315],
  [430, 274], [379, 274], [327, 275], [275, 275], [223, 275], [170, 275],
  [437, 230], [383, 230], [328, 231], [273, 231], [217, 230], [161, 230],
  [447, 182], [388, 180], [329, 181], [270, 180], [212, 180], [151, 180],
]

worldPoints = np.array(world_points, dtype=np.float32)
#MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE

#[u,v] center + 9 Image points
imagePoints = np.array(cam_points, dtype=np.float32)

# Create and train the model
model = LinearRegression()
model.fit(imagePoints, worldPoints)

# Save the model to a file
joblib.dump(model, 'new_linear_regression_model.pkl')
print("Model saved as 'new_linear_regression_model.pkl'")
