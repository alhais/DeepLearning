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

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
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
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
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
        #Z = Input(shape=(256,))
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        [I0, Z0] = self.generator([I,img_B])

        # For the combined model we will only train the generator
        self.discriminator.trainable = true


        # Discriminators determines validity of translated images / condition pairs
        [valid, match] = self.discriminator([I0, Z0])

        self.combined = Model(inputs=[img_A, I, img_B], outputs=[valid, match, I0, Z0])
        self.combined.compile(loss=['mse', 'mse','mae', 'mae'],
                              loss_weights=[1, 1, 100, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        # Image input
        input_EMG = Input(shape=self.img_shape)
        e1 = conv2d(input_EMG, self.gf, bn=False)
        e2 = conv2d(e1, self.gf*2)
        h = Flatten()(e2)
        z0 = Dense(256)(h)

        
        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        h = Flatten()(d7)
        h = Dense(256)(h)
        h = Concatenate()([h, z0])
        h = Reshape((1, 512))(h)
        h = GRU(512, recurrent_initializer="orthogonal")(h)
        h = Reshape((1,1,512))(h)
        d8 = UpSampling2D(size=2)(h)


        h = Conv2DTranspose(filters=self.gf*8, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(d8)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2DTranspose(filters=self.gf*8, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2DTranspose(filters=self.gf*8, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2DTranspose(filters=self.gf*4, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2DTranspose(filters=self.gf*2, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(h)
        h = BatchNormalization(momentum=0.8)(h)
        h = Conv2DTranspose(filters=self.gf, kernel_size=(5),\
        strides=(2,2), padding='SAME', activation='relu')(h)
        h = BatchNormalization(momentum=0.8)(h)
        u7 = UpSampling2D(size=2)(h)

        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model([d0,input_EMG], [output_img,z0])

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        d1 = d_layer(img_A, self.gf, bn=False)
        d2 = d_layer(d1, self.gf*2)
        d3 = d_layer(d2, self.gf*4)
        d4 = d_layer(d3, self.gf*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        Z = Input(shape=(256,))
        h =Reshape((16, 16, 1))(Z)
        h = Concatenate()([validity, h])
        d = Conv2D(self.gf*8, kernel_size=4, strides=2, padding='same')(h)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(self.gf*4, kernel_size=1, strides=1, padding='same')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = BatchNormalization(momentum=0.8)(d)


        match = Conv2D(1, kernel_size=4, strides=2, padding='valid')(d)

        return Model([img_A, Z], [validity, match])

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        match = np.zeros((batch_size,) + (3,3,1))
        match_fake = np.zeros((batch_size,) + (3,3,1))
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
                d_loss_fake = self.discriminator.train_on_batch([fake_A, Z0], [fake, match_fake])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_I, imgs_A, imgs_B], [valid, match, imgs_A, Z0])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 30

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=30, is_testing=True)
        imgs_I = []
        for x in range(imgs_A.shape[0]):
            imgs_I.append(imgs_A[0])
        imgs_I = np.array(imgs_I)
        [fake_A, Z0] = self.generator.predict([imgs_I,imgs_B])
        fake_A = fake_A*0.5
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
    gan.train(epochs=200, batch_size=1, sample_interval=100)
