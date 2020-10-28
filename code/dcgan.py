# Copyright (C) 2020 Andreas Pentaliotis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# DCGAN Module
# Deep convolutional generative adversarial network model.

import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import BatchNormalization, Input, LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation, Dense, Flatten, Reshape
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import utility

plt.rcParams.update({"font.size": 12})


class DCGAN:

  def __init__(self, height, width, depth):
    self._height = height
    self._width = width
    self._depth = depth

    self._data_generator = ImageDataGenerator(zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
                                              rotation_range=10)
    self._generator, self._discriminator, self._adversarial = self._build_models()

  def _build_models(self):
    optimizer = Adam(lr=0.0002, beta_1=0.5)

    # Build the generator and discriminator and compile the discriminator.
    generator = self._build_generator()
    discriminator = self._build_discriminator()
    discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Build and compile the adversarial network.
    discriminator.trainable = False
    noise = Input(shape=(100,))
    fake_image = generator(noise)
    label = discriminator(fake_image)
    adversarial = Model(noise, label)
    adversarial.compile(loss="binary_crossentropy", optimizer=optimizer)

    return generator, discriminator, adversarial

  def _build_discriminator(self):
    input_shape = (self._height, self._width, self._depth)

    # Build the model
    model = Sequential()

    model.add(Conv2D(128, kernel_size=5, strides=2, padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1024, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    return model

  def _build_generator(self):
    # Determine initial dimensions.
    height = int(self._height / 16)
    width = int(self._width / 16)

    # Build the model.
    model = Sequential()

    model.add(Dense(height * width * 1024, input_dim=100))
    model.add(Reshape((height, width, 1024)))

    model.add(Conv2DTranspose(512, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2DTranspose(self._depth, kernel_size=5, strides=2, padding="same"))
    model.add(Activation("tanh"))

    return model

  def _save_training_plots(self, directory_path, discriminator_history_real, discriminator_history_fake,
                           generator_history):
    if not os.path.isdir(directory_path):
      print("Output directory does not exist | Creating directories along directory path")
      os.makedirs(directory_path)

    print("Saving training plots | Directory path: {}".format(directory_path))

    plt.plot([x[1] for x in discriminator_history_real])
    plt.plot([x[1] for x in discriminator_history_fake])
    plt.title("Discriminator training accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Real", "Fake"], loc="upper left")
    plt.savefig("{}/discriminator-training-accuracy".format(directory_path))
    plt.close()

    plt.plot([x[0] for x in discriminator_history_real])
    plt.plot([x[0] for x in discriminator_history_fake])
    plt.title("Discriminator training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Real", "Fake"], loc="upper left")
    plt.savefig("{}/discriminator-training-loss".format(directory_path))
    plt.close()

    plt.plot([x for x in generator_history])
    plt.title("Generator training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("{}/generator-training-loss".format(directory_path))
    plt.close()

  def train(self, images, epochs, batch_size, saving_frequency, output_path):
    batches = int(images.shape[0] / batch_size)
    training_generator = self._data_generator.flow(images, batch_size=int(batch_size / 2))

    discriminator_history_real = []
    discriminator_history_fake = []
    generator_history = []
    for epoch in range(1, epochs + 1):
      discriminator_statistics_real = []
      discriminator_statistics_fake = []
      generator_statistics = []
      for _ in range(batches):
        # Select a mini batch of real images randomly, with size half of batch size. Account for the
        # case where the size of images is not divisible by batch size.
        real_images = training_generator.next()
        if real_images.shape[0] != int(batch_size / 2):
          real_images = training_generator.next()
        real_labels = np.ones((int(batch_size / 2), 1))

        # Generate fake images from noise, with size half of batch size.
        noise = np.random.normal(0, 1, (int(batch_size / 2), 100))
        fake_images = self._generator.predict(noise)
        fake_labels = np.zeros((int(batch_size / 2), 1))

        # Train the discriminator.
        discriminator_statistics_real.append(self._discriminator.train_on_batch(real_images, real_labels))
        discriminator_statistics_fake.append(self._discriminator.train_on_batch(fake_images, fake_labels))

        # Sample data points from the noise distribution, with size of batch size and create
        # real labels for them.
        noise = np.random.normal(0, 1, (batch_size, 100))
        real_labels = np.ones((batch_size, 1))

        # Train the generator.
        generator_statistics.append(self._adversarial.train_on_batch(noise, real_labels))

      discriminator_history_real.append(np.average(discriminator_statistics_real, axis=0))
      discriminator_history_fake.append(np.average(discriminator_statistics_fake, axis=0))
      generator_history.append(np.average(generator_statistics, axis=0))

      # Print the statistics for the current epoch.
      print()
      print("Epoch %d/%d" % (epoch, epochs))
      utility.print_line()
      print("Discriminator: [Loss real: %f | Accuracy real: %.2f%% | Loss fake: %f | Accuracy fake: %.2f%%]"
            % (discriminator_history_real[-1][0], 100 * discriminator_history_real[-1][1],
               discriminator_history_fake[-1][0], 100 * discriminator_history_fake[-1][1]))
      print("Generator: [Loss: %f]" % generator_history[-1])

      if epoch % saving_frequency == 0:
        # Save a sample of fake images, the generator, the discriminator and the training history up
        # to the current epoch.
        saving_directory_path = "{}/epoch-{}".format(output_path, str(epoch))
        images = utility.generate_images(self._generator, 10)
        utility.save(images, saving_directory_path)
        self.save_models(saving_directory_path)
        self._save_training_plots(saving_directory_path, discriminator_history_real, discriminator_history_fake,
                                  generator_history)

  def save_models(self, directory_path):
    if not os.path.isdir(directory_path):
      print("Output directory does not exist | Creating directories along directory path")
      os.makedirs(directory_path)

    print("Saving models | Directory path: {}".format(directory_path))

    self._generator.save("{}/generator.h5".format(directory_path))
    self._discriminator.save("{}/discriminator.h5".format(directory_path))

  def summary(self):
    print()
    print("ADVERSARIAL")
    utility.print_line()
    self._adversarial.summary()
    print()
    print("DISCRIMINATOR")
    utility.print_line()
    self._discriminator.summary()
    print()
    print("GENERATOR")
    utility.print_line()
    self._generator.summary()
