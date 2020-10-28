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

# Generate Module
# Module to generate images using a trained generator.

from keras.models import load_model

import utility

input_arguments = utility.parse_input_arguments(module="generate")
generator = load_model(input_arguments.generator_path)
images = utility.generate_images(generator, input_arguments.image_number)
utility.save(images, input_arguments.output_path)
