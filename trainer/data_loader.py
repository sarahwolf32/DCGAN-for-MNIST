'''Handles loading and processing training data.'''
import os
import tensorflow as tf


def load_image(filepath):
    '''Downloads image
    
    Args:
        filepath: String, either a GCS address, or a local filepath
    Returns:
        Tensor of the image pixel values.
    '''
    f = tf.read_file(filepath)
    image = tf.image.decode_png(f, channels=1)
    return image


def scale_images_range(images):
    '''Scales image values to range [-1, 1]
    Args:
        image: Tensor, of shape [M, H, W, 3], with values in range [0, 255]
    Returns:
        A tensor of shape [M, H, W, 3], with values in range [-1, 1]
    '''
    images = tf.cast(images, dtype=tf.float32)
    images = images / 128.
    images = images - 1.
    return images


def load_dataset(config):

    # find images
    file_pattern = config.data_dir + "/*.png"
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)

    # load images
    dataset = dataset.map(lambda filepath: load_image(filepath))

    # split into batches
    dataset = dataset.repeat()
    dataset = dataset.batch(config.batch_size)

    # scale images to range [-1, 1]
    dataset = dataset.map(lambda images: scale_images_range(images))
    
    # compute num batches
    num_images = len(os.listdir(config.data_dir)) - 1
    num_batches = int(num_images / config.batch_size)

    return dataset, num_batches
