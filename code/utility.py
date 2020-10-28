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

# Utility Module
# Utility functions to run image generation.

import argparse
import os

import cv2 as cv
import numpy as np


def print_line():
  print("-" * 65)


def plot(name, image):
  cv.namedWindow(name, cv.WINDOW_NORMAL)
  cv.imshow(name, image)
  cv.waitKey(0)
  cv.destroyWindow(name)


def read_image(image_path):
  image = cv.imread(image_path, cv.IMREAD_COLOR)
  return image


def load_images(directory_path):
  print("Loading images")

  images = []
  for filename in os.listdir("{}/".format(directory_path)):
    image = read_image("{}/{}".format(directory_path, filename))
    images.append(image)

  return images


def resize(images, height, width):
  images = [cv.resize(x, (height, width), interpolation=cv.INTER_CUBIC) for x in images]
  return images


def normalize(images, min_pixel_value, max_pixel_value):
  images = (images - np.min(images)) / np.ptp(images) * (max_pixel_value - min_pixel_value) + min_pixel_value
  return images


def preprocess(images):
  print("Preprocessing images")

  images = resize(images, height=128, width=128)
  images = np.array(images)
  images = normalize(images, min_pixel_value=-1, max_pixel_value=1)

  return images


def shuffle(images):
  print("Shuffling images")
  np.random.seed(0)
  np.random.shuffle(images)


def generate_images(generator, image_number):
  print("Generating images")

  noise = np.random.normal(0, 1, (image_number, 100))
  images = generator.predict(noise)
  images = normalize(images, min_pixel_value=0, max_pixel_value=255)

  return images


def save(images, directory_path):
  if not os.path.isdir(directory_path):
    print("Output directory does not exist | Creating directories along directory path")
    os.makedirs(directory_path)

  print("Saving images | Directory path: {}".format(directory_path))

  image_name = 0
  for image in images:
    cv.imwrite("{}/{}.jpg".format(directory_path, str(image_name)), image)
    image_name += 1


def handle_input_argument_errors(module, input_arguments):
  if module == "train" and input_arguments.epochs < input_arguments.saving_frequency:
    raise ValueError("value of saving_frequency is greater than value of epochs")
  elif module == "train" and input_arguments.saving_frequency <= 0:
    raise ValueError("value of saving_frequency is not positive")


def parse_input_arguments(module, image_number_choices=range(10, 101, 10), epoch_choices=range(500, 2001, 500),
                          batch_size_choices=range(8, 33, 8)):
  parser = None
  if module == "generate":
    parser = argparse.ArgumentParser(prog="generate", usage="generates images using the provided trained generator")
    parser.add_argument("generator_path", help="directory path to trained generator model")
    parser.add_argument("image_number", type=int, choices=image_number_choices, help="number of images to generate")
    parser.add_argument("output_path", help="directory path to save the generated images")
  elif module == "train":
    parser = argparse.ArgumentParser(prog="train", usage="trains a DCGAN model on the provided image data")
    parser.add_argument("image_path", help="directory path to training image data")
    parser.add_argument("epochs", type=int, choices=epoch_choices, help="number of training epochs")
    parser.add_argument("batch_size", type=int, choices=batch_size_choices,
                        help="training data batch size")
    parser.add_argument("saving_frequency", type=int,
                        help="epoch frequency for saving the training history and the model; should be positive and"
                             " within range of epochs")
    parser.add_argument("output_path", help="directory path to save the output of training")

  input_arguments = parser.parse_args()
  handle_input_argument_errors(module, input_arguments)

  return input_arguments
