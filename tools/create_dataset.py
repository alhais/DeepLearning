import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('/datasets/0.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  edges = cv2.Canny(image,100,200)
  vis = np.concatenate((image, edges), axis=1)
  cv2.imwrite("%d.jpg" % count, vis)     # save frame as JPEG file
  success,image = vidcap.read()
  count += 1
