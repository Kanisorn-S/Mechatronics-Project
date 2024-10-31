import numpy as np
from sklearn.linear_model import LinearRegression
import joblib  # for saving the model
import os

#load camera calibration
savedir="camera_info"
cam_mtx=np.load(os.path.join(savedir,'cam_mtx.npy'))
dist=np.load(os.path.join(savedir, 'dist.npy'))
newcam_mtx=np.load(os.path.join(savedir, 'newcam_mtx.npy'))
roi=np.load(os.path.join(savedir, 'roi.npy'))


#load center points from New Camera matrix
cx=newcam_mtx[0,2]
cy=newcam_mtx[1,2]
fx=newcam_mtx[0,0]

print("cx: "+str(cx)+",cy "+str(cy)+",fx "+str(fx))
# Sample data (replace this with your actual data)
X_center=437
Y_center=515
Z_center=486
worldPoints=np.array([[X_center,Y_center,Z_center],
                       [495,225,524],
                       [720,225,524],
                       [945,225,525],
                       [495,450,491],
                       [720,450,495],
                       [945,450,497],
                       [495,675,467],
                       [720,675,466],
                       [945,675,467]], dtype=np.float32)

#MANUALLY INPUT THE DETECTED IMAGE COORDINATES HERE

#[u,v] center + 9 Image points
imagePoints=np.array([[cx,cy],
                       [303-50, 237],
                       [367-50, 237],
                       [430-50, 235],
                       [303-50, 294],
                       [367-50, 293],
                       [433-50, 293],
                       [298-50, 357],
                       [367-50, 357],
                       [437-50, 357]], dtype=np.float32)

# Create and train the model
model = LinearRegression()
model.fit(imagePoints, worldPoints[:, :2])

# Save the model to a file
joblib.dump(model, 'linear_regression_model.pkl')
print("Model saved as 'linear_regression_model.pkl'")
