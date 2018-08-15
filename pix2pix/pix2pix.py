from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Conv2DTranspose, concatenate, add, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.recurrent import GRU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os
import keras.backend as K

class Pix2Pix():
    def __init__(self):
                  
        def create_adv_loss(discriminator):
            def loss(y_true, y_pred):
                print( y_pred.shape)
                return K.log(1.0 - y_pred) + K.log(1.0 - y_true)
            return loss

        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
    
 
        
        # Configure data loader
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 16
        self.df = 16

        optimizer = Adam(0.0002, 0.5)
        
        

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        adv_loss = create_adv_loss(self.discriminator)
        self.discriminator.compile(loss=adv_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()
        # Input images and their conditioning images
        I = Input(shape=self.img_shape)
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        [I0, Z0] = self.generator([I, img_B])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False


        # Discriminators determines validity of translated images / condition pairs
        [valid, match] = self.discriminator([I0, Z0])

        self.combined = Model(inputs=[img_A, I, img_B], outputs=[valid, match, I0, Z0])
        self.combined.compile(loss=['mse', 'mse','mae','mae'],
                              loss_weights=[1, 1, 100, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        #EMG Decoder
        input_EMG = Input(shape=self.img_shape)
        h = Conv2D(filters=self.gf, kernel_size=(3),\
        strides=(1,1), padding='SAME')(input_EMG)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=self.gf*2, kernel_size=(3),\
        strides=(1,1), padding='SAME')(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = MaxPooling2D(strides=(1, 2), padding='same')(h)
        h = Conv2D(filters=self.gf*4, kernel_size=(3),\
        strides=(1,1), padding='SAME')(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=self.gf*4, kernel_size=(3),\
        strides=(1,1), padding='SAME')(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=self.gf*8, kernel_size=(3),\
        strides=(1,1), padding='SAME')(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = MaxPooling2D(strides=(2, 2), padding='same')(h)
        h = Flatten()(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        z0 = Dense(256)(h)


        # Image input
        input_I = Input(shape=self.img_shape)
        h = Conv2D(filters=self.gf, kernel_size=(5),\
        strides=(2,2), padding='SAME')(input_I)
        h = LeakyReLU(alpha=0.2)(h)
        d0 = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=self.gf*2, kernel_size=(5),\
        strides=(2,2), padding='SAME')(d0)
        h = LeakyReLU(alpha=0.2)(h)
        d1 = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=self.gf*4, kernel_size=(5),\
        strides=(2,2), padding='SAME')(d1)
        h = LeakyReLU(alpha=0.2)(h)
        d2 = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=self.gf*8, kernel_size=(5),\
        strides=(2,2), padding='SAME')(d2)
        h = LeakyReLU(alpha=0.2)(h)
        d3 = BatchNormalization(momentum=0.8)(h)
        h = Flatten()(d3)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        d4 = BatchNormalization(momentum=0.8)(h)
        h = Dense(256)(h)

        h = Concatenate()([h, z0])
        h = Reshape((1, 512))(h)
        h = GRU(512, recurrent_initializer="orthogonal")(h)
        h = Dense(2048)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)

        h = Reshape((2,2,512))(h)

        h = Conv2DTranspose(filters=self.gf*4, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Concatenate()([h, d2])
        h = Conv2DTranspose(filters=self.gf*2, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Concatenate()([h, d1])
        h = Conv2DTranspose(filters=self.gf, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Concatenate()([h, d0])
        h = Conv2DTranspose(filters=8, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=3, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='tanh')(h)
        output_img = UpSampling2D(size=2)(h)

        return Model([input_EMG,input_I], [output_img,z0])

    def build_discriminator(self):
    
        Z0 = Input(shape=(256,))
        # Image input
        I0 = Input(shape=self.img_shape)
        
        h = Conv2D(filters=16, kernel_size=(4),\
        strides=(2,2), padding='SAME')(I0)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=32, kernel_size=(4),\
        strides=(2,2), padding='SAME')(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=64, kernel_size=(4),\
        strides=(2,2), padding='SAME')(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2D(filters=128, kernel_size=(4),\
        strides=(2,2), padding='SAME')(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
        out1 = Conv2D(filters=1, kernel_size=(4),\
        strides=(1,1), padding='SAME')(h)
        h = Conv2D(filters=256, kernel_size=(4),\
        strides=(2,2), padding='SAME')(out1)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)

        d0 = Reshape((1, 1, 256))(Z0)
        #d0 = UpSampling2D(size=4)(d0)

        h = Concatenate()([h, d0])
        h = Conv2D(filters=1, kernel_size=(1),\
        strides=(1,1), padding='SAME')(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization(momentum=0.8)(h)
  
        out2 = Conv2D(filters=1, kernel_size=(1),\
        strides=(2,2), padding='valid')(h)

        return Model([I0,Z0],[out1,out2])

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        match = np.ones((batch_size,) + (1,1,1))
        fake_match = np.ones((batch_size,) + (1,1,1))
        fake = np.zeros((batch_size,) + self.disc_patch)
        

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                #Change for a constant I image
                imgs_I = []
                for x in range(imgs_A.shape[0]):
                    imgs_I.append(imgs_A[0])
                imgs_I = np.array(imgs_I)
                [fake_A, Z0] = self.generator.predict([imgs_I,imgs_B])

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, Z0], [valid, match])
                d_loss_fake = self.discriminator.train_on_batch([fake_A, Z0], [fake, fake_match])
                #d_loss_fake = 0
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_I, imgs_A, imgs_B], [valid, match, imgs_A, Z0])

                elapsed_time = datetime.datetime.now() - start_time
        

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
                    # Plot the progress
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            elapsed_time))

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 30

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=30, is_testing=True)
        imgs_I = []
        for x in range(imgs_A.shape[0]):
            imgs_I.append(imgs_A[0])
        imgs_I = np.array(imgs_I)
        [fake_A, Z0] = self.generator.predict([imgs_I,imgs_B])
        #fake_A = fake_A*0.5
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
        
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        titles = ['EMG MFFC', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i],fontsize=1)
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i), dpi=900)
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=1, sample_interval=250)
