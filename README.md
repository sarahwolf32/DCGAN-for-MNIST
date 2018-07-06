# DCGAN for MNIST

This is an implementation of the paper ['Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks' by Radford, et al](https://arxiv.org/pdf/1511.06434.pdf) on the MNIST dataset of hand-written digits.

## Background

[Generative Adversarial Networks (GANs)](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) are a very interesting class of generative models that were introduced by Ian Goodfellow in 2014. The promise of GANs was that they could be given a set of unlabeled data, and be trained to generate new samples similar to that data. For example, feed the model a large set of images of cats, and it would learn to create new, original images of cats that it had never seen before. 

This is exciting for several reasons. On a practical level, there is far more unlabeled data in the world than there is labeled data. For researchers, it is exciting because it is an example of unsupervised learning. Learning useful representations of unlabeled data has many applications, and as Richard Feynman famously said, "What I cannot create, I do not understand". 

However, early GANs proved to be very difficult to train, especially with complex data like images. In an effort to overcome this, the authors of this paper introduced DCGAN, a GAN specialized for images. From extensive model exploration, they found a set of architectures that work well for image generation.

## How do GANs work?

A GAN pits two models against each other - a 'Generator' (<i>G</i>), that takes a random code and outputs a generated image, and a 'Discriminator' (<i>D</i>), that attempts to distinguish the real training images from the generated ones. 

Let's define some terms:

<i>x<sub>i</sub></i> = <i>i<sup>th</sup></i> real training example

<i>z<sub>i</sub></i> = <i>i<sup>th</sup></i> randomly generated code

<i>G(z<sub>i</sub>)</i> = output of <i>G</i>, an image generated from <i>z<sub>i</sub></i>

<i>D(x<sub>i</sub>)</i> = output of <i>D</i> when fed the real training example, the probability it gives the real example of being real. D wants this to be as high as possible. 

<i>D(G(z<sub>i</sub>))</i> = output of <i>D</i> when fed the generated example, the probability it gives the fake example of being real. <i>D</i> wants this to be as low as possible, while <i>G</i> wants it to be as high as possible.

## The Cost Function(s)

<img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;Loss_{D}&space;=&space;-\frac{1}{m}&space;\sum_{i=1}^{m}&space;log(D(x_{i}))&space;&plus;&space;log(1&space;-&space;D(G(z_{i})))" height="70"/>

<img src="https://latex.codecogs.com/png.latex?\dpi{300}&space;\large&space;Loss_{G}&space;=&space;-\frac{1}{m}&space;\sum_{i&space;=&space;1}^{m}&space;log(D(G(z_{i})))" height="70"/>


## Architectures for Generating Images

The authors of the DCGAN paper found the following architecture changes to be important for successful training:

* All-convolutional nets, with no spatial pooling functions such as maxpool.
* They eliminated the fully connected layers often placed on top of convolutional layers. Only the input and output layers are not convolutional.
* Batch normalization at every layer except the generator output and the discriminator input layers.
* ReLU activation for all layers in the generator except the output layer, which uses tanh. This necessitates a pre-processing step to scale the training images to [-1, 1], the range of tanh, so that the range of outputs from G will match the real examples.
* LeakyReLU activation in the discriminator for all layers, with a leak-slope of 0.2.
* They used the Adam optimizer, with the momentum parameter B1 set to 0.5, and a learning rate of 0.0002. 

This implementation uses all of the above, with the exception that it uses LeakyReLU in the generator, and no fully connected layers at all.

## Results

The generator started producing convincing samples after about 8 epochs. 

Generated digits:

<img src='trained_examples/zero_0.png'> <img src='trained_examples/one_0.png'> <img src='trained_examples/two_0.png'> <img src='trained_examples/three_0.png'> <img src='trained_examples/four_0.png'> <img src='trained_examples/five_0.png'> <img src='trained_examples/six_0.png'> <img src='trained_examples/seven_0.png'> <img src='trained_examples/eight_0.png'> <img src='trained_examples/nine_0.png'>

<img src='trained_examples/zero_1.png'> <img src='trained_examples/one_1.png'> <img src='trained_examples/two_1.png'> <img src='trained_examples/three_1.png'> <img src='trained_examples/four_1.png'> <img src='trained_examples/five_1.png'> <img src='trained_examples/six_1.png'> <img src='trained_examples/seven_1.png'> <img src='trained_examples/eight_1.png'> <img src='trained_examples/nine_1.png'>

<img src='trained_examples/zero_2.png'> <img src='trained_examples/one_2.png'> <img src='trained_examples/two_2.png'> <img src='trained_examples/three_2.png'> <img src='trained_examples/four_2.png'> <img src='trained_examples/five_2.png'> <img src='trained_examples/six_2.png'> <img src='trained_examples/seven_2.png'> <img src='trained_examples/eight_2.png'> <img src='trained_examples/nine_2.png'>

MNIST training digits:

<img src='mnist_examples/zero.png'> <img src='mnist_examples/one.png'> <img src='mnist_examples/two.png'> <img src='mnist_examples/three.png'> <img src='mnist_examples/four.png'> <img src='mnist_examples/five.png'> <img src='mnist_examples/six.png'> <img src='mnist_examples/seven.png'> <img src='mnist_examples/eight.png'> <img src='mnist_examples/nine.png'>


## Trained Model

You can download the trained model [here](https://storage.googleapis.com/gan-training-207705-public-bucket/MNIST-DCGAN-model-2.zip).

To pull generated samples from this model:
1. Unzip the model, and drag it into the project directory
2. cd into the project directory
3. Run ```python -m trainer.task --sample [NUMBER_OF_SAMPLES_TO_OUTPUT]```
4. You can find the resulting images in the trainer/samples folder. 

If you want to store the trained model somewhere else, just include ```--checkpoint-dir [YOUR_PATH]``` in the command.

If you want to output the samples to another location, just include ```--sample-dir [YOUR_PATH]``` in the command.

## Train Your Own

If you want to tweak this code and train your own version from scratch, you can find the main code in trainer/task.py. To train, you will need to:

1. Download the MNIST data [here](https://storage.googleapis.com/gan-training-207705-public-bucket/mnist.npz).
2. cd into the project directory
3. Run ```python -m trainer.task --data-dir [YOUR_PATH_TO_MNIST_DATA]```

The checkpoint models will written to the MNIST-trained-model folder by default, but you can write to another location by including ```--checkpoint-dir [YOUR_CHECKPOINT_PATH]```.

The event file for Tensorboard will be written to trainer/summary by default, but you can write them to another location by including ```--log-dir [YOUR_LOG_DIR_PATH]```

To view training progress in Tensorboard, run ```tensorboard --log-dir=[YOUR_LOG_DIR_PATH]```, and then open ```localhost:6006```.

Check out trainer/train_config.py for other optional settings. 

If you need to stop and re-start training, you can continue training from a checkpoint saved in the MNIST-trained-model folder with the ```--continue-train True``` flag.


## Acknowledgements

The architecture that actually ended up working was based on this [excellent comparison of GANs and DCGANs](https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/) by Hyeonwoo Kang. 








