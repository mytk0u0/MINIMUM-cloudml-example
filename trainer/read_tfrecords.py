import os
import argparse
import tensorflow as tf

BATCH_SIZE = 50

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
args = parser.parse_args()
tfrecord_dir = os.path.join(args.data_dir, 'tfrecords')


def read_tfrecord(filename_queue):
    reader = tf.TFRecordReader()
    key, record_string = reader.read(filename_queue)
    example = tf.parse_single_example(record_string, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_string': tf.FixedLenFeature([], tf.string)
    })

    label = tf.cast(example['label'], tf.int32)

    image = tf.decode_raw(example['image_string'], tf.uint8)
    image = tf.reshape(image, [28, 28, 3])
    image.set_shape([28, 28, 3])
    return image, label


def input_pipeline(filenames):
    filename_queue = tf.train.string_input_producer(filenames)
    image, label = read_tfrecord(filename_queue)
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=BATCH_SIZE,
        min_after_dequeue=10000,
        capacity=10000 + 3 * BATCH_SIZE,
        num_threads=1,
    )
    return image_batch, label_batch


def main():
    tfrecord_path = os.path.join(tfrecord_dir, 'train.tfrecord')
    image_batch, label_batch = input_pipeline([tfrecord_path, ])

    init_op = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        result = sess.run([image_batch, label_batch])
        print(result)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
