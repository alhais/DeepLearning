import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
images_dir = '/content/pix2pix-tensorflow/pix2pix-tensorflow/DeepLearning/DeepLearning/DeepLearning/datasets'
imgList = os.listdir(images_dir)
nImgs = len(imgList)
print('#images = %d' % nImgs)
img = cv.imread(imgList[200],0)
edges = cv.Canny(img,100,200)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
cv.imwrite('first.jpg',edges)

for i in range(nImgs):
  img = cv.imread(imgList[i],0)
  edges = cv.Canny(img,100,200)
  print('%s' % 'edge_'+imgList[i])
  cv.imwrite('edge_'+imgList[i],edges)