import tensorflow as tf
import numpy as np
import time
import os
from train_config import TrainConfig
from train_ops import TrainOps
from StringIO import StringIO
from tensorflow.python.lib.io import file_io

'''
Generates new hand-written digit images based on the MNIST dataset.
Implementation of DCGAN.
'''

GENERATOR_SCOPE = 'generator'
DISCRIMINATOR_SCOPE = 'discriminator'

def access_data(is_local=False):

    # Downloads data into ~/.keras/datasets/mnist.npz, if its not there.
    # Raw data is a tuple of np arrays ((x_train, y_train), (x_test, y_test))
    if is_local:
        mnist = tf.keras.datasets.mnist.load_data()
        x_train, x_test = mnist[0][0], mnist[1][0]
        return x_train, x_test
    else:
        f = StringIO(file_io.read_file_to_string('gs://gan-training-207705_bucket2/mnist.npz'))
        mnist = np.load(f)  
        x_train, x_test = mnist['x_train'], mnist['x_test']
        return x_train, x_test


def load_data():

    x_train, x_test = access_data()

    # We do not need the labels, so we will gather all examples x.
    # Return a numpy array of shape (M, 28, 28)
    num_train, num_test = x_train.shape[0], x_test.shape[0]
    M = num_train + num_test
    x_all = np.zeros((M, 28, 28))
    x_all[:num_train, :, :] = x_train
    x_all[num_train:, :, :] = x_test
    print("finished load_data!")
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
    print("finished data_tensor!")
    return X


def load_dataset():
    numpy_data = load_data()
    np.random.shuffle(numpy_data)

    batch_size = 128
    num_batches = int(np.ceil(numpy_data.shape[0] / float(batch_size)))

    X = data_tensor(numpy_data)
    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.batch(batch_size)
    print "finished load_dataset()!"
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
        deconv_5 = tf.layers.conv2d_transpose(
            lrelu_4, 
            filters=1,
            kernel_size=[4,4],
            strides=[2,2],
            padding='same',
            activation=tf.nn.tanh,
            kernel_initializer=initializer,
            name='layer5')
        output = tf.identity(deconv_5, name='generated_images')

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
        loss_d = tf.identity(-tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg)), name='loss_d')
        loss_g = tf.identity(-tf.reduce_mean(tf.log(Dg)), name='loss_g')
        return loss_d, loss_g

def increment(variable, sess):
    sess.run(tf.assign_add(variable, 1))
    new_val = sess.run(variable)
    return new_val


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
    train_g = optimizer_g.minimize(loss_g, var_list=g_vars, name='train_g')
    train_d = optimizer_d.minimize(loss_d, var_list = d_vars, name='train_d')

    return train_d, train_g, loss_d, loss_g, generated_images


def save_model(checkpoint_dir, session, step, saver):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_name = checkpoint_dir + '/model-' + str(step) + '.cptk'
    saver.save(session, model_name, global_step=step)
    print("saved model!")

def create_training_ops():

    # create a placeholders
    images_holder = tf.placeholder(tf.float32, shape=[None, 64, 64, 1], name='images_holder')
    Z_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, 100], name='z_holder')

    # get trainers
    train_d, train_g, loss_d, loss_g, generated_images = trainers(images_holder, Z_holder)

    # initialize variables
    global_step_var = tf.Variable(0, name='global_step')
    epoch_var = tf.Variable(0, name='epoch')
    batch_var = tf.Variable(0, name='batch')

    # prepare summaries
    loss_d_summary_op = tf.summary.scalar('Discriminator_Loss', loss_d)
    loss_g_summary_op = tf.summary.scalar('Generator_Loss', loss_g)
    images_summary_op = tf.summary.image('Generated_Image', generated_images, max_outputs=1)
    training_images_summary_op = tf.summary.image('Training_Image', images_holder, max_outputs=1)
    summary_op = tf.summary.merge_all()


def train(sess, ops, config):
    
    writer = tf.summary.FileWriter(config.event_filename, graph=tf.get_default_graph())

    # prepare data
    dataset, num_batches = load_dataset()
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    saver = tf.train.Saver()

    epoch = sess.run(ops.epoch_var)
    batch = sess.run(ops.batch_var)
    global_step = sess.run(ops.global_step_var)

    # loop over epochs
    while epoch < config.num_epochs:
        sess.run(iterator.initializer)
        print("starting epoch loop! " + str(epoch))

        # loop over batches
        while batch < num_batches:
            print("starting batch loop! " + str(batch))

            # inputs
            images = sess.run(next_batch)
            Z = np.random.normal(0.0, 1.0, size=[images.shape[0], 1, 1, 100])

            # run session
            feed_dict = {'images_holder:0': images, 'z_holder:0': Z}
            sess.run(ops.train_d, feed_dict=feed_dict)
            sess.run(ops.train_g, feed_dict=feed_dict)

            # logging
            if global_step % config.log_freq == 0:
                summary = sess.run(ops.summary_op, feed_dict=feed_dict)
                writer.add_summary(summary, global_step=global_step)

                loss_d_val = sess.run(ops.loss_d, feed_dict=feed_dict)
                loss_g_val = sess.run(ops.loss_g, feed_dict=feed_dict)
                print("epoch: " + str(epoch) + ", batch " + str(batch))
                print("G loss: " + str(loss_g_val))
                print("D loss: " + str(loss_d_val))

            # saving
            if global_step % config.checkpoint_freq == 0:
                save_model(config.checkpoint_dir, sess, global_step, saver)

            global_step = increment(ops.global_step_var, sess)
            batch = increment(ops.batch_var, sess)

        epoch = increment(ops.epoch_var, sess)
        sess.run(tf.assign(ops.batch_var, 0))
        batch = sess.run(ops.batch_var)

    sess.close()


def begin_training(config):
    create_training_ops()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ops = TrainOps()
    ops.populate(sess)
    train(sess, ops, config)


def continue_training(config):
    with tf.Session() as sess:

        # load stored graph into current graph
        graph_filename = str(tf.train.latest_checkpoint(config.checkpoint_dir)) + '.meta'
        saver = tf.train.import_meta_graph(graph_filename)

         # restore variables into graph
        saver.restore(sess, tf.train.latest_checkpoint(config.checkpoint_dir))
        
        # load operations 
        ops = TrainOps()
        ops.populate(sess)

        # train
        train(sess, ops, config)
    

def sample(config):
    num_samples = int(config.sample)
    Z = np.random.normal(0.0, 1.0, size=[num_samples, 1, 1, 100])

    with tf.Session() as sess:

        # load stored graph into current graph
        graph_filename = str(tf.train.latest_checkpoint(config.checkpoint_dir)) + '.meta'
        saver = tf.train.import_meta_graph(graph_filename)

        # restore variables into graph
        saver.restore(sess, tf.train.latest_checkpoint(config.checkpoint_dir))
        
        # load operations 
        ops = TrainOps()
        ops.populate(sess)

        # sample
        images = sess.run(ops.generated_images, feed_dict={'z_holder:0': Z})
        images = images + 1.
        images = images * 128.
        for i in range(images.shape[0]):
            image = images[i]
            img_tensor = tf.image.encode_png(image)
            img_name = 'samples/sample_' + str(i) + '.png'
            output_file = open(img_name, 'wb+')
            output_data = sess.run(img_tensor)
            output_file.write(output_data)
            output_file.close()


# Main
def main(_):
    # parse arguments
    config = TrainConfig()
    config.populate()
    
    # train
    if config.should_continue:
        continue_training(config)
    elif config.sample > 0:
        sample(config)
    else:
        begin_training(config)
    

if __name__ == '__main__':
    tf.app.run()
 
