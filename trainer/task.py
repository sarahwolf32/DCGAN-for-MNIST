import tensorflow as tf
import numpy as np
import time
import os
import io
from io import StringIO
from .train_config import TrainConfig
from .train_ops import TrainOps
from tensorflow.python.lib.io import file_io
from .data_loader import *
from .networks import *

'''
Generates new hand-written digit images based on the MNIST dataset.
Implementation of DCGAN.
'''


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

    return train_d, train_g, loss_d, loss_g, generated_images, Dx, Dg


def save_model(checkpoint_dir, session, step, saver):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_name = checkpoint_dir + '/model-' + str(step) + '.cptk'
    saver.save(session, model_name, global_step=step)
    print("saved model!")

def create_training_ops():

    # create a placeholders
    images_holder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='images_holder')
    Z_holder = tf.placeholder(tf.float32, shape=[None, 1, 1, 100], name='z_holder')

    # get trainers
    train_d, train_g, loss_d, loss_g, generated_images, Dx, Dg = trainers(images_holder, Z_holder)

    # initialize variables
    global_step_var = tf.Variable(0, name='global_step')
    epoch_var = tf.Variable(0, name='epoch')
    batch_var = tf.Variable(0, name='batch')

    # prepare summaries
    loss_d_summary_op = tf.summary.scalar('Discriminator_Loss', loss_d)
    loss_g_summary_op = tf.summary.scalar('Generator_Loss', loss_g)
    dx_summary_op = tf.summary.scalar('D(x)', tf.reduce_mean(Dx))
    dg_summary_op = tf.summary.scalar('D(g)', tf.reduce_mean(Dg))

    images_summary_op = tf.summary.image('Generated_Image', generated_images, max_outputs=3)
    training_images_summary_op = tf.summary.image('Training_Image', images_holder, max_outputs=3)
    summary_op = tf.summary.merge_all()


def train(sess, ops, config):
    
    writer = tf.summary.FileWriter(config.summary_dir, graph=tf.get_default_graph())

    # prepare data
    dataset, num_batches = load_dataset(config)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    saver = tf.train.Saver()

    epoch = sess.run(ops.epoch_var)
    batch = sess.run(ops.batch_var)
    global_step = sess.run(ops.global_step_var)

    # loop over epochs
    while epoch < config.num_epochs:
        sess.run(iterator.initializer)

        # loop over batches
        while batch < num_batches:

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

def load_session(config):
    sess = tf.Session()

    # load stored graph into current graph
    graph_filename = str(tf.train.latest_checkpoint(config.checkpoint_dir)) + '.meta'
    saver = tf.train.import_meta_graph(graph_filename)

    # restore variables into graph
    saver.restore(sess, tf.train.latest_checkpoint(config.checkpoint_dir))
        
    # load operations 
    ops = TrainOps()
    ops.populate(sess)
    return sess, ops

def continue_training(config):
    sess, ops = load_session(config)
    train(sess, ops, config)    

def sample(config):
    sess, ops = load_session(config)
    num_samples = int(config.sample)
    Z = np.random.normal(0.0, 1.0, size=[num_samples, 1, 1, 100])

    # get images
    images = sess.run(ops.generated_images, feed_dict={'z_holder:0': Z})
    images = images + 1.
    images = images * 128.

    # write to disk
    for i in range(images.shape[0]):
        image = images[i]
        img_tensor = tf.image.encode_png(image)
        img_name = config.sample_dir + '/sample_' + str(i) + '.png'
        output_file = open(img_name, 'wb+')
        output_data = sess.run(img_tensor)
        output_file.write(output_data)
        output_file.close()
    

# Main
def main(_):
    # parse arguments
    config = TrainConfig(local=True)
    
    # train
    if config.should_continue:
        continue_training(config)
    elif config.sample > 0:
        sample(config)
    else:
        begin_training(config)
    

if __name__ == '__main__':
    tf.app.run()
 
