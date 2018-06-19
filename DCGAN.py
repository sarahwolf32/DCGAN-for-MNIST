import tensorflow as tf
import numpy as np
import time
import argparse
import os

'''
Generates new hand-written digit images based on the MNIST dataset.
Implementation of DCGAN.
'''

GENERATOR_SCOPE = 'generator'
DISCRIMINATOR_SCOPE = 'discriminator'

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
    '''
    numpy_data: (M, 28, 28), values in range [0, 255]
    returns: tensor of images shaped [M, 64, 64, 1], with values in range [-1, 1]
    '''

    # Turn numpy array into a tensor X of shape [M, 28, 28, 1].
    X = tf.constant(numpy_data)
    X = tf.reshape(X, [-1, 28, 28, 1])

    # resize images to 64x64
    X = tf.image.resize_images(X, [64, 64])

    # The data is currently in a range [0, 255].
    # Transform data to have a range [-1, 1].
    # We do this to match the range of tanh, the activation on the generator's output layer.
    X = X / 128.
    X = X - 1.
    return X


def load_dataset():
    numpy_data = load_data()
    np.random.shuffle(numpy_data)

    batch_size = 128
    num_batches = int(np.ceil(numpy_data.shape[0] / float(batch_size)))

    X = data_tensor(numpy_data)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(batch_size)
    return dataset, num_batches


def generator(Z, initializer):
    '''
    Takes an argument Z, an [M, 1, 1, 100] tensor of random numbers.
    Returns an operation that creates a generated image of shape [None, 64, 64, 1]
    '''

    with tf.variable_scope(GENERATOR_SCOPE):

        # Layer 1
        deconv_1 = tf.layers.conv2d_transpose(
            Z, 
            filters=1024, 
            kernel_size=[4,4], 
            strides=[1,1], 
            padding='valid',
            kernel_initializer=initializer, 
            name='layer1')
        norm_1 = tf.layers.batch_normalization(deconv_1)
        lrelu_1 = tf.nn.leaky_relu(norm_1)

        # Layer 2
        deconv_2 = tf.layers.conv2d_transpose(
            lrelu_1, 
            filters=512, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer,
            name='layer2')
        norm_2 = tf.layers.batch_normalization(deconv_2)
        lrelu_2 = tf.nn.leaky_relu(norm_2)

        # Layer 3
        deconv_3 = tf.layers.conv2d_transpose(
            lrelu_2,
            filters=256, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer,
            name='layer3')
        norm_3 = tf.layers.batch_normalization(deconv_3)
        lrelu_3 = tf.nn.leaky_relu(norm_3)

        # Layer 4
        deconv_4 = tf.layers.conv2d_transpose(
            lrelu_3,
            filters=128, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer,
            name='layer4')
        norm_4 = tf.layers.batch_normalization(deconv_4)
        lrelu_4 = tf.nn.leaky_relu(norm_4)

        # Output Layer
        output = tf.layers.conv2d_transpose(
            lrelu_4, 
            filters=1,
            kernel_size=[4,4],
            strides=[2,2],
            padding='same',
            activation=tf.nn.tanh,
            kernel_initializer=initializer,
            name='layer5')

        # Generated images of shape [M, 64, 64, 1]
        return output


# Discriminator
def discriminator(images, initializer, reuse=False):
    '''
    Takes an image as an argument [None, 64, 64, 1].
    Returns an operation that gives the probability of that image being 'real' [None, 1]
    '''

    with tf.variable_scope(DISCRIMINATOR_SCOPE, reuse=reuse):

        # Layer 1 
        conv_1 = tf.layers.conv2d(
            images, 
            filters=128, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            activation=tf.nn.leaky_relu,
            kernel_initializer=initializer, 
            name='layer1')

        # Layer 2 
        conv_2 = tf.layers.conv2d(
            conv_1, 
            filters=256, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer,
            name='layer2')
        norm_2 = tf.layers.batch_normalization(conv_2)
        lrelu_2 = tf.nn.leaky_relu(norm_2)

        # Layer 3 
        conv_3 = tf.layers.conv2d(
            lrelu_2, 
            filters=512, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer, 
            name='layer3')
        norm_3 = tf.layers.batch_normalization(conv_3)
        lrelu_3 = tf.nn.leaky_relu(norm_3)

        # Layer 4
        conv_4 = tf.layers.conv2d(
            lrelu_3, 
            filters=1024, 
            kernel_size=[4,4], 
            strides=[2,2], 
            padding='same', 
            kernel_initializer=initializer, 
            name='layer4')
        norm_4 = tf.layers.batch_normalization(conv_4)
        lrelu_4 = tf.nn.leaky_relu(norm_4)

        # Layer 5
        conv_5 = tf.layers.conv2d(lrelu_4, filters=1, kernel_size=[4,4], strides=[1,1], padding='valid')
        sigmoid_5 = tf.nn.sigmoid(conv_5)
        output = tf.reshape(sigmoid_5, [-1, 1])

        return output

# loss
def loss(Dx, Dg):
    '''
    Dx = Probabilities assigned by D to the real images, [M, 1]
    Dg = Probabilities assigned by D to the generated images, [M, 1]
    '''
    with tf.variable_scope('loss'):
        loss_d = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg))
        #loss_g = tf.reduce_mean(tf.log(1. - Dg))
        loss_g = -tf.reduce_mean(tf.log(Dg))
        return loss_d, loss_g


# Training
def trainers(images_holder, Z_holder):

    # forward pass
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
    generated_images = generator(Z_holder, weights_initializer)
    Dx = discriminator(images_holder, weights_initializer, False)
    Dg = discriminator(generated_images, weights_initializer, True)

    # compute losses
    loss_d, loss_g = loss(Dx, Dg)

    # optimizers
    optimizer_g = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    optimizer_d = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

    # backprop
    g_vars = tf.trainable_variables(scope=GENERATOR_SCOPE)
    d_vars = tf.trainable_variables(scope=DISCRIMINATOR_SCOPE)
    train_g = optimizer_g.minimize(loss_g, var_list=g_vars)
    train_d = optimizer_d.minimize(loss_d, var_list = d_vars)

    return train_d, train_g, loss_d, loss_g, generated_images

def save_model(checkpoint_dir, session, step, saver):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_name = checkpoint_dir + '/model-' + str(step) + '.cptk'
    saver.save(session, model_name)
    print("saved model!")


def train(event_filename, log_freq, num_epochs, checkpoint_freq, checkpoint_dir):

    # create a placeholders
    images_holder = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
    Z_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, 100])

    # get trainers
    train_d, train_g, loss_d, loss_g, generated_images = trainers(images_holder, Z_holder)
    init = tf.global_variables_initializer()

    # prepare summaries
    loss_d_summary_op = tf.summary.scalar('Discriminator Loss', loss_d)
    loss_g_summary_op = tf.summary.scalar('Generator Loss', loss_g)
    images_summary_op = tf.summary.image('Generated Image', generated_images, max_outputs=1)
    training_images_summary_op = tf.summary.image('Training Image', images_holder, max_outputs=1)
    summary_op = tf.summary.merge_all()
    writer = tf.summary.FileWriter(event_filename, graph=tf.get_default_graph())

    saver = tf.train.Saver()

    # begin session
    sess = tf.Session()
    sess.run(init)

    # dataset
    dataset, num_batches = load_dataset()
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    # loop over epochs
    global_step = 0
    for epoch in range(num_epochs):
        sess.run(iterator.initializer)

        # loop over batches
        for i in range(num_batches):
            #start_time = time.time()

            # train
            images = sess.run(next_batch)
            Z = np.random.normal(0.0, 1.0, size=[images.shape[0], 1, 1, 100])

            # run session
            feed_dict = {images_holder: images, Z_holder: Z}
            sess.run(train_d, feed_dict=feed_dict)
            sess.run(train_g, feed_dict=feed_dict)

            # logging
            if global_step % log_freq == 0:
                summary = sess.run(summary_op, feed_dict=feed_dict)
                writer.add_summary(summary, global_step=global_step)

                loss_d_val = sess.run(loss_d, feed_dict=feed_dict)
                loss_g_val = sess.run(loss_g, feed_dict=feed_dict)
                print("epoch: " + str(epoch) + ", batch " + str(i))
                print("G loss: " + str(loss_g_val))
                print("D loss: " + str(loss_d_val))

            # saving
            if global_step % checkpoint_freq == 0:
                save_model(checkpoint_dir, sess, global_step, saver)
            global_step += 1

            # monitor time
            #end_time = time.time()
            #run_time = end_time - start_time 
            #print("batch-time: " + str(run_time))

    sess.close()

# Main
if __name__ == '__main__':

    # parse arguments
    # python DCGAN.py --event-file-dir test_1
    parser = argparse.ArgumentParser()
    parser.add_argument('--event-file-dir', help='Folder name for Tensorboard output', required=False)
    parser.add_argument('--log-freq', help='n, where progress is logged every n steps', required=False)
    parser.add_argument('--num-epochs', help='number of epochs to train', required=False)
    args = parser.parse_args()

    # unwrap optional arguments
    event_filename = args.event_file_dir or 'summary'
    log_freq = args.log_freq or 2
    num_epochs = args.num_epochs or 15

    # train
    train(
        event_filename=event_filename,
        log_freq=log_freq,
        num_epochs=num_epochs,
        checkpoint_freq=1,
        checkpoint_dir='checkpoints'
    )








"""
# Temporary Tests
     # Test generator
    # create a [M, 100] constant
    # run it through the generator
    # output should be correct shape
print ("testing my generator!")
weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
M = 5
Z = tf.random_uniform([M, 1, 1, 100])
layer1, layer2, layer3, layer4, o = generator(Z, weights_initializer)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
layer1_val, layer2_val, layer3_val, layer4_val, o_val = sess.run([layer1, layer2, layer3, layer4, o])
print("input: " + str(Z.shape))
print("Layer1: " + str(layer1_val.shape))
print("Layer2: " + str(layer2_val.shape))
print("Layer3: " + str(layer3_val.shape))
print("Layer4: " + str(layer4_val.shape))
print("out: " + str(o.shape))

"""
"""
# Test discriminator
    # create a [M, 32, 32, 1] constant
    # run it through the discriminator
    # output should be of the correct shape
print("testing discriminator!")
weights_initializer = tf.truncated_normal_initializer(stddev=0.02)
M = 5
images = tf.random_uniform([M, 64, 64, 1])
D = discriminator(images, weights_initializer)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    out = sess.run(D)
    print("got an out! Should be shape [M, 1]")
    print("out.shape:")
    print(out.shape)
    print(out) 
"""
"""
# Test loss
    # create two [M, 1] constants
    # evaluate losses
    # should get back two floats
print("testing loss!")
Dx = tf.random_uniform([M, 1])
Dg = tf.random_uniform([M, 1])
with tf.Session() as sess:
    loss_d, loss_g = sess.run(loss(Dx, Dg))
    print("loss_d = ", str(loss_d))
    print("loss_g = ", str(loss_g)) """


""" 
print("done loading data!")

sess = tf.Session()

d = sess.run(X)
print("data shape:")
print(d.shape) 
"""