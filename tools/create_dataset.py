from python_speech_features import mfcc
import shutil
import csv
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

print(cv2.__version__)
os.makedirs(os.getcwd() + '/DeepLearning/pix2pix/datasets/facades/train')
os.makedirs(os.getcwd() + '/DeepLearning/pix2pix/datasets/facades/test')

emg0,emg1,emg2,emg3,emg4,emg5,emg6,emg7 = ([] for i in range(8))

with open(os.getcwd() + '/DeepLearning/datasets/emg0.csv', newline='') as csvfile:
  reader = csv.reader(csvfile, delimiter=',', quotechar='|')
  next(reader, None)
  for row in reader:
      emg0.append(int(row[1]))
      emg1.append(int(row[2]))
      emg2.append(int(row[3]))
      emg3.append(int(row[4]))
      emg4.append(int(row[5]))
      emg5.append(int(row[6]))
      emg6.append(int(row[7]))
      emg7.append(int(row[8]))

lenght = len(emg0)     
emg_image = np.zeros([8,lenght],dtype=np.uint8)
for x in range(lenght):
  emg_image[0,x] = emg0[x] + 128
  emg_image[1,x] = emg1[x] + 128
  emg_image[2,x] = emg2[x] + 128
  emg_image[3,x] = emg3[x] + 128
  emg_image[4,x] = emg4[x] + 128
  emg_image[5,x] = emg5[x] + 128
  emg_image[6,x] = emg6[x] + 128
  emg_image[7,x] = emg7[x] + 128

plt.figure()
plt.title('CSV EMG Image')
plt.imshow(emg_image,interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect=300)
plt.colorbar()
plt.gca().grid(False)
cv2.imwrite('EMG_image.jpg',emg_image)



signal = emg_image[0,:]

mfcc_single_image = mfcc(signal,samplerate = 250,winlen = 0.035, winstep = 0.010, numcep = 10)#numcep = 20 was experimental
mfcc_image = np.empty((mfcc_single_image.shape[0], mfcc_single_image.shape[1]),dtype=np.uint8)
mfcc_image = (mfcc_single_image+25)*255/50
for x in range(8):
  signal = emg_image[x,:]
  mfcc_single_image = mfcc(signal,samplerate = 250,winlen = 0.035, winstep = 0.010, numcep = 10)#numcep = 20 was experimental
  mfcc_single_image = (mfcc_single_image+25)*255/50
  mfcc_image = np.append(mfcc_image,mfcc_single_image , axis=-1)

#print(signal.shape)
#print(mfcc_image.shape)

ig, ax = plt.subplots()
mfcc_data = np.swapaxes(mfcc_image, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
ax.set_title('MFCC')
#Showing mfcc_data
plt.show()
cv2.imwrite('MFCC_image.jpg',mfcc_data)


#from google.colab import files
#import os
#files.download('EMG_image.jpg')
mcc_length = mfcc_image.shape[0]
vidcap = cv2.VideoCapture(os.getcwd() + '/DeepLearning/datasets/0.mp4')
n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))


#Calculate the distribution of MFCC time to the corresponding frames
winSize = int(mcc_length/n_frames)
step = 1
print("window size is %d"%winSize)

# Pre-compute number of chunks to emit
numOfChunks = int(((mfcc_data.shape[1]-winSize)/step)+1)
chuncks = np.zeros([mfcc_data.shape[0],winSize,numOfChunks],dtype=np.uint8)

# Do the work
count = 0
for i in range(0,numOfChunks*step,step):
    chuncks[:,:,i] =  mfcc_data[:,i:i+winSize]
    
#print(chuncks[:,:,0])
#print(mfcc_data[:,0:0+winSize])
#print(mcc_length,n_frames)
count = 0
for x in range(int(n_frames/2)):
  success,image = vidcap.read()
  if success:
    cv2.imwrite(os.getcwd() + '/DeepLearning/datasets/facades/train/' + "%d_emg.jpg" % count, chuncks[:,:,x])
    cv2.imwrite(os.getcwd() + '/DeepLearning/datasets/facades/train/' + "%d.jpg" % count, image)
    #print(os.getcwd() + '/DeepLearning/datasets/train/' + "%d.jpg created"% count)
    count += 1
count = 0
for x in range(int(n_frames/2)):
  success,image = vidcap.read()
  if success:
    cv2.imwrite(os.getcwd() + '/DeepLearning/pix2pix/datasets/facades/test/' + "%d_emg.jpg" % count, chuncks[:,:,x])
    cv2.imwrite(os.getcwd() + '/DeepLearning/pix2pix/datasets/facades/test/' + "%d.jpg" % count, image)
    #print(os.getcwd() + '/DeepLearning/datasets/train/' + "%d.jpg created"% count)
    count += 1
    
#from google.colab import files
#import os
#files.download(os.getcwd() + '/DeepLearning/datasets/train/' + "100_emg.jpg")
