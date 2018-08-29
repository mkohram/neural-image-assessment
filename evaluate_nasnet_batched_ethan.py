import numpy as np
import argparse
from path import Path

from keras.models import Model
from keras.layers import Dense, Dropout
# from keras.preprocessing.image import load_img, img_to_array
# import matplotlib.pyplot as plt
import tensorflow as tf
import glob
# import subprocess
import pickle
# import PIL as p

from utils.nasnet import NASNetMobile, preprocess_input
from utils.score_utils import mean_score, std_score

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s')
logger = logging.getLogger()

IMAGE_SIZE = 224

def parse_data(filename):
    '''
    Loads the image file without any augmentation. Used for validation set.
    Args:
        filename: the filename from the record
        scores: the scores from the record
    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.reshape(image, tf.shape(image))
    #https://github.com/kumasento/tensorflow/commit/4796e6535a4245cf3ae935ca715b71723244142e
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image

def data_generator(batchsize):
    '''
    Creates a python generator that loads the AVA dataset images without random data
    augmentation and generates numpy arrays to feed into the Keras model for training.
    Args:
        batchsize: batchsize for validation set
    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        val_dataset = tf.data.Dataset().from_tensor_slices(IMGS)
        val_dataset = val_dataset.map(parse_data)

        val_dataset = val_dataset.batch(batchsize)
        val_iterator = val_dataset.make_initializable_iterator()

        val_batch = val_iterator.get_next()

        sess.run(val_iterator.initializer)

        while True:
            try:
                X_batch = sess.run(val_batch)
                yield X_batch
            except:
                val_iterator = val_dataset.make_initializable_iterator()
                sess.run(val_iterator.initializer)
                val_batch = val_iterator.get_next()

                X_batch = sess.run(val_batch)
                yield X_batch

with tf.device('/CPU:0'):
    IMGS = glob.glob('data/*')
    base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/nasnet_weights.h5')

    print('starting nn')
    logger.info(f'Start')
    scores = model.predict_generator(data_generator(500), 10, verbose=2)
    print(scores.shape)
    logger.info(f'Done')
