# Image Generation Deep Learning

This repository contains a deep learning system that performs image generation using a deep convolutional generative adversarial network (DCGAN). The dataset url and the associated article url are provided below.

Dataset url: https://www.dropbox.com/sh/isslk5zkp9ekqtc/AABTfbuLYRID6NhDvq1Vi7Hha?dl=0

Article url: https://medium.com/@yvanscher/using-gans-to-create-monsters-for-your-game-c1a3ece2f0a0

## Installation

It is recommended to install conda and then create an environment for the system using the ```environment.yaml``` file. A suggestion on how to install the system and activate the environment is provided below.

```bash
git clone https://github.com/anpenta/image-generation-deep-learning.git
cd image-generation-deep-learning
conda env create -f environment.yaml
conda activate image-generation-deep-learning
```

## Running the system

To run the system for training you can provide commands through the terminal using the ```train``` module. An example is given below.

```bash
python3 train.py ./pokemon-data 1000 8 100 ./output
```
This will train a model using data from the ```./pokemon-data``` directory for 1000 epochs with a batch size of 8 and save the training plots and the trained model every 100 epochs in the ```./output``` directory. An example of how to see the parameters for training is provided below.

```bash
python3 train.py --help
```

To run the system for generating images you can provide commands through the terminal using the ```generate``` module. An example is given below.


```bash
python3 generate.py ./generator.h5 50 ./output
```
This will use the generator from the ```./generator.h5``` file to generate 50 images and save them in the ```./output``` directory. An example of how to see the parameters for generation is provided below.

```bash
python3 generate.py --help
```

## Results

As an example, below are the training results we get after training a model for 1460 epochs. The generator does not generate sensible images every time, so I chose five of the best ones. In general, the training is not smooth and you may have to restart training if the discriminator outperforms the generator for many consecutive epochs or vice versa.

<p float="left">
<img src=./training-results/0.jpg height="128" width="128">
<img src=./training-results/1.jpg height="128" width="128">
<img src=./training-results/2.jpg height="128" width="128">
<img src=./training-results/3.jpg height="128" width="128">
<img src=./training-results/4.jpg height="128" width="128">
</p>

<p float="left">
<img src=./training-results/discriminator-training-accuracy.png height="320" width="420">
<img src=./training-results/discriminator-training-loss.png height="320" width="420">
<img src=./training-results/generator-training-loss.png height="320" width="420">
</p>

## Sources
* Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
