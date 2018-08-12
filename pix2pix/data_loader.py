import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import cv2
from numpy import random

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        pathA =sorted(glob.glob('./datasets/%s/%s/A/*' % (self.dataset_name, data_type)), key=os.path.getmtime)
        pathB =sorted(glob.glob('./datasets/%s/%s/B/*' % (self.dataset_name, data_type)), key=os.path.getmtime)
        
        seq_array = list(range(len(pathA)))
        rd_array = random.choice(seq_array, size=len(pathA), replace=False)   
    
        batch_images_A = []
        batch_images_B = []
        for x in range(batch_size):
            batch_images_A.append(pathA[rd_array[x]])
            batch_images_B.append(pathB[rd_array[x]])


        imgs_A = []
        imgs_B = []
        for img_path in batch_images_A:
            img_A = self.imread(img_path)
            img_A = scipy.misc.imresize(img_A, self.img_res)
            imgs_A.append(img_A)
        imgs_A = np.array(imgs_A)/127.5 - 1.
            
        for img_path in batch_images_B:
            img_B = self.imread(img_path)
            #img_B = scipy.misc.imresize(img_B, self.img_res)
            img_B = cv2.resize(img_B.astype('uint8'),(self.img_res), interpolation=cv2.INTER_NEAREST)
            imgs_B.append(img_B)

        imgs_B = np.array(imgs_B)/127.5 - 1.
        
        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        pathA =sorted(glob.glob('./datasets/%s/%s/A/*' % (self.dataset_name, data_type)), key=os.path.getmtime)
        pathB =sorted(glob.glob('./datasets/%s/%s/B/*' % (self.dataset_name, data_type)), key=os.path.getmtime)

        self.n_batches = int(len(pathA) / batch_size)

        for i in range(self.n_batches-1):
            batch = pathA[i*batch_size:(i+1)*batch_size]
            imgs_A = []
            for img in batch:
                img_A = self.imread(img)
                img_A = scipy.misc.imresize(img_A, self.img_res)                  
                imgs_A.append(img_A)
                
            imgs_A = np.array(imgs_A)/127.5 - 1.
            
            batch = pathB[i*batch_size:(i+1)*batch_size]
            imgs_B = []
            for img in batch:
                img_B = self.imread(img)
                #img_B = scipy.misc.imresize(img_B, self.img_res)
                img_B = cv2.resize(img_B.astype('uint8'),(self.img_res), interpolation=cv2.INTER_NEAREST)         
                imgs_B.append(img_B)
                
            imgs_B = np.array(imgs_B)/127.5 - 1.
            
            yield imgs_A, imgs_B
            



    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
