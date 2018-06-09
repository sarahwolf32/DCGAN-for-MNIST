import tensorflow as tf
import numpy as np

'''
Generates new hand-written digit images based on the MNIST dataset.
Implementation of DCGAN.
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


# Load data
""" numpy_data = load_data()
X = data_tensor(numpy_data)
print("done loading data!")

sess = tf.Session()

d = sess.run(X)
print("data shape:")
print(d.shape) """




# Generator
def generator(Z):
    '''
    Takes Z as an argument, a [M, 100] tensor of random numbers.
    Returns an operation that creates an image of shape [None, 32, 32, 1]
    '''

    # Input Layer
    input_fc = tf.contrib.layers.dense(
        input_norm, 
        num_outputs=4*4*256, 
        activation=tf.nn.relu)
    input_reshape = tf.reshape(input_fc, [None, 4, 4, 256])

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
        kernel_size=[32,32],
        padding='SAME',
        activation=tf.nn.tanh)

    return output


# Discriminator
'''
Takes an image as an argument [None, 32, 32, 1].
Returns an operation that gives the probability of that image being 'real' [None, 1]

Conv1
    16 filters
    kernal_size = [4, 4]
    stride = [2, 2]
    padding = 'SAME'
    LeakyReLU, slope = 0.2

Conv2
    32 filters
    kernal_size = [4, 4]
    stride = [2, 2]
    padding = 'SAME'
    Batch Norm
    LeakyReLU, slope = 0.2

Conv3
    64 filters
    kernal_size = [4, 4]
    stride = [2, 2]
    padding = 'SAME'
    Batch Norm
    LeakyReLU, slope = 0.2

Output
    Flatten
    FC - 1 node
    Sigmoid

'''

# Training
'''
Loss(D) = -(1/m) SUM_OVER_M[ log(D(x)) + log(1 - D(G(z)))]
Loss(G) = (1/m) SUM_OVER_M[ log(1 - D(G(z)))]

Mini-batch size 128
Mini-batch stochastic gradient descent
Initialize weights from zero-centered normal distribution, 0.02 standard deviation
AdamOptimizer => learning_rate = 0.0002, B1 = 0.5
'''