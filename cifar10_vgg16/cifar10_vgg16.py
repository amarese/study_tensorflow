# -*- coding: utf-8 -*-
import tensorflow as tf
import re
import numpy as np
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

# model data
data_dict = None
var_dict = {}

# config
number_classes = 10
input_size = 32 * 32 * 3
training_epochs = 1000
max_epoch = 10000
batch_size = 1000
test_size = 1000


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


'''
get variables from data_dict, if not exist, initialize it and put into data_dict
'''


def get_var(initial_value, name, idx, var_name):
    if data_dict is not None and name in data_dict:
        value = data_dict[name][idx]
    else:
        value = initial_value

    var = tf.Variable(value, name=var_name)
    var_dict[(name, idx)] = var
    return var


def get_conv_var(filter_width, filter_height, in_channels, out_channels, width, height, name):
    weights = get_var(tf.random_normal([filter_width, filter_height, in_channels, out_channels], 0, 0.001), name, 0,
                      name + "_weights")
    biases = get_var(tf.random_normal([out_channels], 0, 0.001), name, 1, name + "_biases")
    return weights, biases


def conv_layer(input_tensor, out_channels, width, height, name):
    in_channels = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights, biases = get_conv_var(3, 3, in_channels, out_channels, width, height, name)
        conv = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        return tf.nn.relu(conv, name=name + "_activation")


def get_fc_var(in_size, out_size, name):
    weights = get_var(tf.random_normal([in_size, out_size], 0, 0.001), name, 0, name + "_weights")
    biases = get_var(tf.random_normal([out_size], 0, 0.001), name, 1, name + "_biases")
    return weights, biases


def fc_layer(input_tensor, out_size, name):
    in_size = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights, biases = get_fc_var(in_size, out_size, name)
        fc = tf.matmul(input_tensor, weights)
        fc = tf.nn.bias_add(fc, biases)
        return tf.nn.relu(fc, name=name + "_activation")


def get_model(X):
    keep_prob = 0.5
    # Block 1
    conv1 = conv_layer(X, 64, 32, 32, "conv1")
    conv2 = conv_layer(conv1, 64, 32, 32, "conv2")
    mp_1 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool1')
    # Block 2
    conv3 = conv_layer(mp_1, 128, 16, 16, "conv3")
    conv4 = conv_layer(conv3, 128, 16, 16, "conv4")
    mp_2 = tf.nn.max_pool(conv4, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool2')
    # Block 3
    conv5 = conv_layer(mp_2, 256, 8, 8, "conv5")
    conv6 = conv_layer(conv5, 256, 8, 8, "conv6")
    conv7 = conv_layer(conv6, 256, 8, 8, "conv7")
    mp_3 = tf.nn.max_pool(conv7, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool3')
    # Block 4
    conv8 = conv_layer(mp_3, 512, 4, 4, "conv8")
    conv9 = conv_layer(conv8, 512, 4, 4, "conv9")
    conv10 = conv_layer(conv9, 512, 4, 4, "conv10")
    mp_4 = tf.nn.max_pool(conv10, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool4')
    # Block 5
    conv11 = conv_layer(mp_4, 512, 2, 2, "conv11")
    conv12 = conv_layer(conv11, 512, 2, 2, "conv12")
    conv13 = conv_layer(conv12, 512, 2, 2, "conv13")
    mp_5 = tf.nn.max_pool(conv13, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool5')

    flattened_shape = np.prod([s.value for s in mp_5.get_shape()[1:]])
    flatten = tf.reshape(mp_5, [-1, flattened_shape], name="flatten")

    fc1 = fc_layer(flatten, 4096, "fc1")
    fc1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = fc_layer(fc1, 4096, "fc2")
    fc2 = tf.nn.dropout(fc2, keep_prob)
    fc3 = fc_layer(fc2, number_classes, "fc3")
    softmax = tf.nn.softmax(fc3, name="softmax");
    return tf.identity(softmax, "output");


def get_cost(model, result, name):
    with tf.variable_scope(name):
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=result))
        cost = tf.reduce_mean(-tf.reduce_sum(result * tf.log(model + 1e-7), axis=1))
        return cost


def get_accuracy(model, result, name):
    with tf.variable_scope(name):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model, 1), tf.argmax(result, 1)), tf.float32))
        return accuracy


def variable_summaries(cost, accuracy, name):
    with tf.name_scope(name):
        tf.summary.scalar('cost', cost)
        tf.summary.scalar('accuracy', accuracy)
        return tf.summary.merge_all()


def get_epoch_number(model_path):
    found_num = re.search(r'\d+', model_path)
    if found_num:
        checkpoint_id = int(found_num.group(0))
        return checkpoint_id


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = load_data()
    y_train_one_hot = dense_to_one_hot(y_train, number_classes)
    y_test_one_hot = dense_to_one_hot(y_test, number_classes)
    return (x_train, y_train_one_hot), (x_test, y_test_one_hot)


def main():
    # load cifar-10
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()

    # placeholder
    X = tf.placeholder(tf.float32, [None, 32, 32, 3], name="input")
    Y = tf.placeholder(tf.float32, [None, number_classes], name="labels")

    # model
    model = get_model(X)

    # cost and optimizer
    cost = get_cost(model, Y, "cost")
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    # accuracy
    accuracy = get_accuracy(model, Y, "accuracy")

    summary = variable_summaries(cost, accuracy, "summaries")

    saver = tf.train.Saver()
    # training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("./checkpoint")
        if checkpoint:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            current_epoch = get_epoch_number(checkpoint.model_checkpoint_path) + 1
        else:
            current_epoch = 0

        writer = tf.summary.FileWriter('result', sess.graph)

        for epoch in range(current_epoch, current_epoch + training_epochs):
            total_batch = int(len(x_train) / batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
                sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            saver.save(sess, "./checkpoint/model.ckpt", epoch)

            cb, ab = sess.run([cost, accuracy], feed_dict={X: batch_xs, Y: batch_ys});
            x_test_batch, y_test_batch = next_batch(test_size, x_test, y_test)
            c, a, s = sess.run([cost, accuracy, summary], feed_dict={X: x_test_batch, Y: y_test_batch});
            print('epoch :', '%04d' % (epoch + 1), ' batch cost :', '{:.9f}'.format(cb), ' accuracy :',
                  '{:.9f}'.format(ab), ' test cost :', '{:.9f}'.format(c), ' accuracy :', '{:.9f}'.format(a))
            writer.add_summary(s, epoch)
            if epoch > max_epoch:
                break;

        print("training done")

        # evaluate
        print("accuracy: ", accuracy.eval(session=sess, feed_dict={X: x_test_batch, Y: y_test_batch}))

        # save results
        minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["output"])

        tf.train.write_graph(minimal_graph, 'result', 'graph.proto', as_text=False)
        tf.train.write_graph(minimal_graph, 'result', 'graph.txt', as_text=True)


if __name__ == '__main__':
    main()
