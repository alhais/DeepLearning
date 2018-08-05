import shutil
import os
import cv2
import numpy as np
print(cv2.__version__)
vidcap = cv2.VideoCapture(os.getcwd() + '/DeepLearning/datasets/0.mp4')
success,image = vidcap.read()
count = 0
success = True
os.makedirs(os.getcwd() + '/DeepLearning/datasets/train')
os.makedirs(os.getcwd() + '/pix2pix-tensorflow/facades/train')
while success:
  edges = cv2.Canny(image,100,200)
  color_edges = cv2.cvtColor(edges, 8)
  vis = np.concatenate((image, color_edges), axis=1)
  resized_image = cv2.resize(vis, (512, 256)) 
  cv2.imwrite(os.getcwd() + '/DeepLearning/datasets/train/' + "%d.jpg" % count, resized_image)     # save frame as JPEG file
  success,image = vidcap.read()
  count += 1
shutil.copytree(os.getcwd() + '/DeepLearning/datasets/train/',os.getcwd() + '/pix2pix-tensorflow/facades/train')
