from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Conv2DTranspose
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
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(I,img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

         #@title Image Encoder
        input_img = Input(shape=self.img_shape)
        # 1 Conv,BN,ReLU
        h = Conv2D(filters=64, kernel_size=(5),\
        strides=(2,2), padding='SAME')(input_img)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 2 Conv,BN,ReLU
        h = Conv2D(filters=128, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 3 Conv,BN,ReLU
        h = Conv2D(filters=256, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 4 Conv,BN,ReLU
        h = Conv2D(filters=512, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 5 FC,BN,ReLU
        h = Flatten()(h)
        h = Dense(512)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 6 FC
        image = Dense(256)(h)

        #model = Model(input_img, image)
        #model.summary()

        #@title EEG Encoder
        input_EMG = Input(shape=self.img_shape)
        # 1 Conv,BN,ReLU
        h = Conv2D(filters=64, kernel_size=(5),\
        strides=(2,2), padding='SAME')(input_EMG)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 2 Conv,BN,ReLU
        h = Conv2D(filters=128, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 3 Conv,BN,ReLU
        h = Conv2D(filters=256, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 4 Conv,BN,ReLU
        h = Conv2D(filters=512, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 5 FC,BN,ReLU
        h = Flatten()(h)
        h = Dense(512)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        # 6 FC,BN,ReLU
        EMG = Dense(256)(h)


        #model = Model(input_EMG, EMG)
        #model.summary()

        #Concate output, is the input of GRU
        GRU_input = concatenate([image, EMG])
        GRU_input = Reshape((1, 512))(GRU_input)
        GRU_output = GRU(512, recurrent_initializer="orthogonal")(GRU_input)


        #Decoder
        #1 FC,BN,ReLU
        h = Dense(25088)(GRU_output)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        #2 Deconv,BN,ReLU
        h = Reshape((4, 4, 1568))(h)
        h = Conv2DTranspose(filters=512, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        #3 Deconv,BN,ReLU
        h = Conv2DTranspose(filters=256, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        #4 Deconv,BN,ReLU
        h = Conv2DTranspose(filters=128, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        #5 Deconv,BN,ReLU
        h = Conv2DTranspose(filters=32, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        #Output Deconv,Tanh
        h = Conv2DTranspose(filters=3, kernel_size=(5),\
        strides=(2,2), padding='SAME')(h)
        DecoderOut = Activation('tanh')(h)
       
        
        return Model([input_img, input_EMG ], DecoderOut)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

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
        r, c = 3, 3

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=200, batch_size=1, sample_interval=200)
