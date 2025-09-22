import os
import glob
import numpy as np
import cv2
from datetime import datetime
import pickle
import torch

data_file = "../data/refined/data128.pkl"
f = open(data_file, "rb")
data = pickle.load(f)
videos = data["videos"]
video = videos[35]
print(video.shape)
video = np.transpose(video, (1, 2, 3, 0))
print(video[0].shape)
for i in range(0, 15):
    print(video[i])
    cv2.imshow("image", video[i])
    cv2.waitKey()
