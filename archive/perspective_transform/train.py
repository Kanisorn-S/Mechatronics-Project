import numpy as np
from sklearn.linear_model import LinearRegression
import joblib  # for saving the model
import os

worldPoints=np.array([ [495,225,524],
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
imagePoints=np.array([ [303-50, 237],
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
