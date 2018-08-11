import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os, glob

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        batch_images_A =sorted(glob.glob('./datasets/%s/%s/A/*' % (self.dataset_name, data_type)), key=os.path.getmtime)
        batch_images_B =sorted(glob.glob('./datasets/%s/%s/B/*' % (self.dataset_name, data_type)), key=os.path.getmtime)


        #batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images_A:
            img_A = self.imread(img_path)
            img_A = scipy.misc.imresize(img_A, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)

            imgs_A.append(img_A)
        imgs_A = np.array(imgs_A)/127.5 - 1.
            
        for img_path in batch_images_B:
            img_B = self.imread(img_path)
            img_B = scipy.misc.imresize(img_B, self.img_res)

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_B = np.fliplr(img_B)

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
                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)                    
                imgs_A.append(img_A)
                
            imgs_A = np.array(imgs_A)/127.5 - 1.
            
            batch = pathB[i*batch_size:(i+1)*batch_size]
            imgs_B = []
            
            for i in range(self.n_batches-1):
            batch = pathB[i*batch_size:(i+1)*batch_size]
            imgs_B = []
            for img in batch:
                img_B = self.imread(img)
                img_B = scipy.misc.imresize(img_B, self.img_res)
                if not is_testing and np.random.random() > 0.5:
                        img_B = np.fliplr(img_B)          
                imgs_B.append(img_B)
                
            imgs_B = np.array(imgs_B)/127.5 - 1.
            
            yield imgs_A, imgs_B
            



    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
