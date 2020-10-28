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

# Train Module
# Module to train a DCGAN model on image data.

import dcgan
import utility

input_arguments = utility.parse_input_arguments(module="train")
images = utility.load_images(input_arguments.image_path)
images = utility.preprocess(images)
utility.shuffle(images)

dcgan = dcgan.DCGAN(images.shape[1], images.shape[2], images.shape[3])
dcgan.summary()

dcgan.train(images, epochs=input_arguments.epochs, batch_size=input_arguments.batch_size,
            saving_frequency=input_arguments.saving_frequency, output_path=input_arguments.output_path)
