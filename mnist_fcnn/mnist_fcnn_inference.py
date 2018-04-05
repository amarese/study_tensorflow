# -*- coding: utf-8 -*-
import random

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def create_graph():
    with tf.gfile.FastGFile("./result/graph.proto", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def main():
    # load mnist data
    mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)

    create_graph()

    with tf.Session() as sess:
        X = sess.graph.get_tensor_by_name("input:0")
        Y = sess.graph.get_tensor_by_name('output:0')
        r = random.randrange(mnist.test.num_examples)
        print('Label : ', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
        print('Prediction :', sess.run(tf.argmax(Y, 1), {X: mnist.test.images[r:r + 1]}))


if __name__ == '__main__':
    main()
