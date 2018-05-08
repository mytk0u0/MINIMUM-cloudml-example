import os
import argparse
import tensorflow as tf
import numpy as np
import skimage.color
from tensorflow.examples.tutorials.mnist import input_data


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfexample(image, label):
    image_string = image.tostring()
    return tf.train.Example(features=tf.train.Features(feature={
        'label': int64_feature(label),
        'image_string': bytes_feature(image_string)
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    mnist_data_dir = os.path.join(args.data_dir, 'MNIST_data')
    tfrecord_dir = os.path.join(args.data_dir, 'tfrecords')

    mnist_data = input_data.read_data_sets(mnist_data_dir, seed=71)

    for data, name in zip(mnist_data, ('train', 'validation', 'test')):
        tfrecord_path = os.path.join(tfrecord_dir, '{}.tfrecord'.format(name))

        images = data.images.reshape(-1, 28, 28)
        labels = data.labels
        images = np.array([skimage.color.gray2rgb(i) for i in images])
        images = (images * 255).astype(np.uint8)

        assert images.shape[1:] == (28, 28, 3)

        with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
            for image, label in zip(images, labels):
                example = convert_to_tfexample(image, label)
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    main()
