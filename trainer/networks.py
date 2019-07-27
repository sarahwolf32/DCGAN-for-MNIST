import tensorflow as tf

GENERATOR_SCOPE = 'generator'
DISCRIMINATOR_SCOPE = 'discriminator'

def generator(Z, initializer):
    '''
    Takes an argument Z, an [M, 1, 1, 100] tensor of random numbers.
    Returns an operation that creates a generated image of shape [None, 32, 32, 1]
    '''

    with tf.variable_scope(GENERATOR_SCOPE):

        # Layer 1 -> [None, 4, 4, 128]
        deconv_1 = tf.layers.conv2d_transpose(
            Z, 
            filters=128, 
            kernel_size=[4,4], 
            strides=[1,1], 
            padding='valid',
            kernel_initializer=initializer, 
            name='layer1')
        norm_1 = tf.layers.batch_normalization(deconv_1)
        lrelu_1 = tf.nn.leaky_relu(norm_1)

        # Layer 2 -> [None, 8, 8, 64]
        deconv_2 = tf.layers.conv2d_transpose(
            lrelu_1, 
            filters=64, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer,
            name='layer2')
        norm_2 = tf.layers.batch_normalization(deconv_2)
        lrelu_2 = tf.nn.leaky_relu(norm_2)

        # Layer 3 -> [None, 16, 16, 32]
        deconv_3 = tf.layers.conv2d_transpose(
            lrelu_2,
            filters=32, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer,
            name='layer3')
        norm_3 = tf.layers.batch_normalization(deconv_3)
        lrelu_3 = tf.nn.leaky_relu(norm_3)

        # Layer 4 -> [None, 32, 32, 1]
        deconv_4 = tf.layers.conv2d_transpose(
            lrelu_3, 
            filters=1,
            kernel_size=[4,4],
            strides=[2,2],
            padding='same',
            activation=tf.nn.tanh,
            kernel_initializer=initializer,
            name='layer4')
        output = tf.identity(deconv_4, name='generated_images')

        # Generated images of shape [M, 32, 32, 1]
        return output


# Discriminator
def discriminator(images, initializer, reuse=False):
    '''
    Takes an image as an argument [None, 32, 32, 1].
    Returns an operation that gives the probability of that image being 'real' [None, 1]
    '''

    with tf.variable_scope(DISCRIMINATOR_SCOPE, reuse=reuse):

        # Layer 1 -> [None, 16, 16, 32]
        conv_1 = tf.layers.conv2d(
            images, 
            filters=32, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            activation=tf.nn.leaky_relu,
            kernel_initializer=initializer, 
            name='layer1')

        # Layer 2 -> [None, 8, 8, 64]
        conv_2 = tf.layers.conv2d(
            conv_1, 
            filters=64, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer,
            name='layer2')
        norm_2 = tf.layers.batch_normalization(conv_2)
        lrelu_2 = tf.nn.leaky_relu(norm_2)

        # Layer 3  -> [None, 4, 4, 128]
        conv_3 = tf.layers.conv2d(
            lrelu_2, 
            filters=128, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer, 
            name='layer3')
        norm_3 = tf.layers.batch_normalization(conv_3)
        lrelu_3 = tf.nn.leaky_relu(norm_3)

        # Layer 5 -> [None, 1, 1, 1]
        conv_4 = tf.layers.conv2d(lrelu_3, filters=1, kernel_size=[4,4], strides=[1,1], padding='valid')
        sigmoid_4 = tf.nn.sigmoid(conv_4)
        output = tf.reshape(sigmoid_4, [-1, 1])

        return output
