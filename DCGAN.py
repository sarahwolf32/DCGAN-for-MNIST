import tensorflow as tf
import numpy as np

'''
Generates new hand-written digit images based on the MNIST dataset.
Implementation of DCGAN.

Issues:
    - Slightly unsure if I'm putting the batch norm layers in the right place.
        - For a batch_norm to belong to a 'layer', should it go at the beginning, 
        between the linearity and activation, or at the end?
        - Every layer except the generator output and the discriminator input.
'''


def load_data():

    # Downloads data into ~/.keras/datasets/mnist.npz, if its not there.
    # Raw data is a tuple of np arrays ((x_train, y_train), (x_test, y_test))
    mnist = tf.keras.datasets.mnist.load_data()

    # We do not need the labels, so we will gather all examples x.
    # Return a numpy array of shape (M, 28, 28)
    x_train, x_test = mnist[0][0], mnist[1][0]
    num_train, num_test = x_train.shape[0], x_test.shape[0]
    M = num_train + num_test
    x_all = np.zeros((M, 28, 28))
    x_all[:num_train, :, :] = x_train
    x_all[num_train:, :, :] = x_test
    return x_all


def data_tensor(numpy_data):

    # Pad the images to make them 32x32, a more convenient size for this model architecture.
    x = np.zeros((numpy_data.shape[0], 32, 32))
    x[:, 2:30, 2:30] = numpy_data

    # The data is currently in a range [0, 255].
    # Transform data to have a range [-1, 1].
    # We do this to match the range of tanh, the activation on the generator's output layer.
    x = x / 128.
    x = x - 1.

    # Turn numpy array into a tensor X of shape [M, 32, 32, 1].
    X = tf.constant(x)
    X = tf.reshape(X, [-1, 32, 32, 1])
    return X


def generator(Z):
    '''
    Takes an argument Z, a [M, 100] tensor of random numbers.
    Returns an operation that creates a generated image of shape [None, 32, 32, 1]
    '''

    # Input Layer
    input_fc = tf.layers.dense(Z, 4*4*256, activation=tf.nn.relu)
    input_reshape = tf.reshape(input_fc, [-1, 4, 4, 256])

    # Layer 1
    norm_1 = tf.layers.batch_normalization(input_reshape)
    deconv_1 = tf.layers.conv2d_transpose(
        norm_1, 
        filters=32, 
        kernel_size=[5,5], 
        strides=[2,2], 
        padding='SAME',
        activation=tf.nn.relu)

    # Layer 2
    norm_2 = tf.layers.batch_normalization(deconv_1)
    deconv_2 = tf.layers.conv2d_transpose(
        norm_2, 
        filters=32, 
        kernel_size=[5,5], 
        strides=[2,2], 
        padding='SAME', 
        activation=tf.nn.relu)

    # Layer 3
    norm_3 = tf.layers.batch_normalization(deconv_2)
    deconv_3 = tf.layers.conv2d_transpose(
        norm_3,
        filters=16, 
        kernel_size=[5,5], 
        strides=[2,2], 
        padding='SAME', 
        activation=tf.nn.relu)

    # Output Layer
    output_norm = tf.layers.batch_normalization(deconv_3)
    output = tf.layers.conv2d_transpose(
        output_norm, 
        filters=1,
        kernel_size=[32,32],
        padding='SAME',
        activation=tf.nn.tanh)

    # Generated images of shape [M, 32, 32, 1]
    return output


# Discriminator
def discriminator(images):
    '''
    Takes an image as an argument [None, 32, 32, 1].
    Returns an operation that gives the probability of that image being 'real' [None, 1]
    '''

    # Layer 1 => [M, 16, 16, 16]
    conv_1 = tf.layers.conv2d(
        images, 
        filters=16, 
        kernel_size=[4,4], 
        strides=[2,2], 
        padding='SAME', 
        activation=tf.nn.leaky_relu)

    # Layer 2 => [M, 8, 8, 32]
    norm_2 = tf.layers.batch_normalization(conv_1) 
    conv_2 = tf.layers.conv2d(
        norm_2, 
        filters=32, 
        kernel_size=[4,4], 
        strides=[2,2], 
        padding='SAME', 
        activation=tf.nn.leaky_relu)

    # Layer 3 => [M, 4, 4, 64]
    norm_3 = tf.layers.batch_normalization(conv_2)
    conv_3 = tf.layers.conv2d(
        norm_3, 
        filters=64, 
        kernel_size=[4,4], 
        strides=[2,2], 
        padding='SAME', 
        activation=tf.nn.leaky_relu)

    # Output layer
    output_norm = tf.layers.batch_normalization(conv_3)
    output_flat = tf.layers.flatten(output_norm) # => [M, 4*4*64]
    output = tf.layers.dense(output_flat, units=1, activation=tf.nn.sigmoid) # => [M, 1]

    return output


# Training
'''
Loss(D) = -(1/m) SUM_OVER_M[ log(D(x)) + log(1 - D(G(z)))]
Loss(G) = (1/m) SUM_OVER_M[ log(1 - D(G(z)))]

Mini-batch size 128
Mini-batch stochastic gradient descent
Initialize weights from zero-centered normal distribution, 0.02 standard deviation
AdamOptimizer => learning_rate = 0.0002, B1 = 0.5
'''


# Main
'''
numpy_data = load_data()
X = data_tensor(numpy_data)
'''

# Test discriminator
    # create a [M, 32, 32, 1] constant
    # run it through the discriminator
    # output should be of the correct shape
print("testing discriminator!")
M = 5
images = tf.random_uniform([M, 32, 32, 1])
D = discriminator(images)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    out = sess.run(D)
    print("got an out!")
    print("out.shape:")
    print(out.shape)
    print(out)


""" 
print("done loading data!")

sess = tf.Session()

d = sess.run(X)
print("data shape:")
print(d.shape) 
"""