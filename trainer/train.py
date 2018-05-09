import os
import argparse
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 50

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--max_steps', type=int, required=True)
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


def input_fn(filenames, repeat_count=-1):
    """featuresとlabelsを返すようにtrain_input_fnを定義する"""
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
    return input_fn([path])


def eval_input_fn():
    path = os.path.join(tfrecord_dir, 'validation.tfrecord')
    return input_fn([path], repeat_count=1)


def json_serving_input_fn():
    inputs = {
        'image': tf.placeholder(tf.uint8, shape=[None, 28, 28, 3])
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


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


def model_fn(features, labels, mode):
    """
    tf.estimator.Estimatorに引き渡す。この関数は、eatures, labels,
    mode, (params) の順に引数を受け取るような関数として定義する必要がある。

    paramsを指定した場合、tf.estimator.Estimatorの引数として
    与えられたparamsを受け取ることができる。
    """

    logits = network(features['image'], mode)

    predictions = {
        # classesは予測と評価に使う
        "classes": tf.argmax(input=logits, axis=1),
        # probabilitiesは予測結果をグラフに出力するのに使う
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    export_outputs = {
        'predict_output': tf.estimator.export.PredictOutput(predictions)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:  # 予測時
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, export_outputs=export_outputs)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:  # 学習時
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(  # 評価時
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    raise KeyError  # 無効なmodeが指定された場合


def main():
    train_spec = tf.estimator.TrainSpec(
        train_input_fn,
        max_steps=args.max_steps,
    )

    exporter = tf.estimator.FinalExporter(
        'mnist',
        json_serving_input_fn
    )
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=100,
        exporters=[exporter],
        name='mnist-eval'
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=args.model_dir)

    tf.estimator.train_and_evaluate(
        estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
