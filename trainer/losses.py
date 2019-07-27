'''A new, eager execution version of main train file, to replace task.py'''
import tensorflow as tf


def loss(Dx, Dg):
    '''
    Dx = Probabilities assigned by D to the real images, [M, 1]
    Dg = Probabilities assigned by D to the generated images, [M, 1]
    '''
    loss_d = -tf.reduce_mean(tf.log(Dx) + tf.log(1. - Dg))
    loss_g = -tf.reduce_mean(tf.log(Dg))
    return loss_d, loss_g


def pre_averaged_loss(Dx, Dg):
    '''
    This loss attempts to reduce spikiness and sudden collapse by averaging Dx and Dg
    *before* they are plugged into the loss equation.
    '''
    mean_dx = tf.reduce_mean(Dx)
    mean_dg = tf.reduce_mean(Dg)
    loss_d = -(tf.log(mean_dx) + tf.log(1. - mean_dg))
    loss_g = -(tf.log(mean_dg))
    return loss_d, loss_g