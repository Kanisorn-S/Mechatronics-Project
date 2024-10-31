import cv2 as cv
import torch
import matplotlib.pyplot as plt
from enum import Enum
import os
import numpy as np

class ModelType(Enum):
  DPT_LARGE = "DPT_LARGE"
  DPT_Hybrid = "DPT_Hybrid"
  MIDAS_SMALL = "MiDaS_small"

class Midas():
  def __init__(self, modelType:ModelType=ModelType.DPT_LARGE):
    self.midas = torch.hub.load("isl-org/MiDaS", modelType.value)
    self.modelType = modelType
    
  def useCUDA(self):
    if torch.cuda.is_available():
      print('Using CUDA')
      self.device = torch.device("cuda")
    else:
      print('Using CPU')
      self.device = torch.device("cpu")
    self.midas.to(self.device)
    self.midas.eval()
    
  def transform(self):
    print("Transform")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if self.modelType.value == "DPT_LARGE" or self.modelType.value == "DPT_Hybrid":
      self.transform = midas_transforms.dpt_transform
    else:
      self.transform = midas_transforms.small_transform
      
  def predict(self, frame):
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    input_batch = self.transform(img).to(self.device)
    with torch.no_grad():
      prediction = self.midas(input_batch)
      prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
      ).squeeze()
    depthMap = prediction.cpu().numpy()
    depthMap = cv.normalize(depthMap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    depthMap = cv.applyColorMap(depthMap, cv.COLORMAP_INFERNO)
    return depthMap
  
  def predict_point(self, frame, point):
    u, v = point
    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    input_batch = self.transform(img).to(self.device)
    with torch.no_grad():
      prediction = self.midas(input_batch)
      prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
      ).squeeze()
    depthMap = prediction.cpu().numpy()
    depthMap = cv.normalize(depthMap, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    return depthMap[u][v] 

  def livePredict(self):
    print("Starting webcam (press q to quit)...")
    capObj = cv.VideoCapture(1)
    while True:
      ret, frame = capObj.read()
      depthMap = self.predict(frame)
      combined = np.hstack((frame, depthMap))
      cv.imshow("Combined", combined)
      if cv.waitKey(1) & 0xFF == ord('q'):
        break
    capObj.release()
    cv.destroyAllWindows()

def run(modelType: ModelType):
  midasObj = Midas(modelType)
  midasObj.useCUDA()
  midasObj.transform()
  midasObj.livePredict()
  
if __name__ == "__main__":
  run(ModelType.MIDAS_SMALL)
    
    