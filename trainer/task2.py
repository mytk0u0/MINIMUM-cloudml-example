import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)

args = parser.parse_args()

with tf.gfile.Open(args.path) as f:
    print(f.readlines())
