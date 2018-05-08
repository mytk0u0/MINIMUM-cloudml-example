import os
import argparse
import tensorflow as tf

BATCH_SIZE = 50

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--num_epochs', type=int, required=True)
args = parser.parse_args()
tfrecord_dir = os.path.join(args.data_dir, 'tfrecords')


# データセットの準備

def parse_tfrecord(record_string):
    example = tf.parse_single_example(record_string, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_string': tf.FixedLenFeature([], tf.string)
    })

    label = tf.cast(example['label'], tf.int32)

    image = tf.decode_raw(example['image_string'], tf.uint8)
    image = tf.reshape(image, [28, 28, 3])
    image.set_shape([28, 28, 3])
    return image, label


def batch_fn(filenames, repeat_count=-1):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.repeat(count=repeat_count)  # デフォルトは無限回
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    features = {"image": images}
    return features, labels


def train_input_fn():
    path = os.path.join(tfrecord_dir, 'train.tfrecord')
    return batch_fn([path])


def eval_input_fn():
    path = os.path.join(tfrecord_dir, 'validation.tfrecord')
    return batch_fn([path], repeat_count=1)


# ネットワークの準備

def preprocess(image):
    return tf.to_float(image) * 2 / 255 - 1


def network(images, mode):
    images = tf.map_fn(preprocess, images, tf.float32)

    input_layer = tf.reshape(images, [-1, 28, 28, 3])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(
        inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    return logits


# モデルの作成

def model_fn(features, labels, mode):
    """Model function for CNN."""

    logits = network(features['image'], mode)

    predictions = {
        # classesは予測と評価に使う
        "classes": tf.argmax(input=logits, axis=1),
        # probabilitiesは予測結果をグラフに出力するのに使う
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:  # 予測時
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:  # 学習時
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
        labels=labels,
        predictions=predictions["classes"]
    )}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(  # 評価時
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    raise KeyError  # 無効なmodeが指定された場合


def main():
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=args.model_dir)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    model.train(
        input_fn=train_input_fn, steps=args.num_epochs, hooks=[logging_hook])

    eval_results = model.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
