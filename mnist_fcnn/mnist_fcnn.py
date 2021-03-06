# -*- coding: utf-8 -*-
import tensorflow as tf
import re
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework.graph_util import convert_variables_to_constants

data_dict = None
var_dict = {}

number_classes = 10
input_size = 28 * 28
training_epochs = 10
max_epoch = 10000
batch_size = 100


def get_var(initial_value, name, idx, var_name):
    if data_dict is not None and name in data_dict:
        value = data_dict[name][idx]
    else:
        value = initial_value

    var = tf.Variable(value, name=var_name)
    var_dict[(name, idx)] = var
    return var


def get_fc_var(in_size, out_size, name):
    weights = get_var(tf.truncated_normal([in_size, out_size], 0, 0.001), name, 0, name + "_weights")
    biases = get_var(tf.truncated_normal([out_size], 0, 0.001), name, 1, name + "_biases")
    return weights, biases


def fc_layer(bottom, in_size, out_size, name):
    with tf.variable_scope(name):
        weights, biases = get_fc_var(in_size, out_size, name)
        x = tf.reshape(bottom, [-1, in_size])
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
        return fc


def get_model(X):
    fc = fc_layer(X, input_size, number_classes, "fc")
    return tf.nn.softmax(fc, name="output");


def get_cost(model, result, name):
    with tf.variable_scope(name):
        cost = tf.reduce_mean(-tf.reduce_sum(result * tf.log(model), axis=1))
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

def main():
    # load mnist data
    mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)

    # placeholder
    X = tf.placeholder(tf.float32, [None, input_size], name="input")
    Y = tf.placeholder(tf.float32, [None, number_classes], name="labels")

    # model
    model = get_model(X)

    # cost and optimizer
    cost = get_cost(model, Y, "cost")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

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
            avg_cost = 0
            total_batch = int(mnist.train.num_examples / batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                c, _, s = sess.run([cost, optimizer, summary], feed_dict={X: batch_xs, Y: batch_ys})
                avg_cost += c / total_batch
            writer.add_summary(s, epoch)
            saver.save(sess, "./checkpoint/model.ckpt", epoch)

            print('epoch :', '%04d' % (epoch + 1), ' cost :', '{:.9f}'.format(avg_cost))
            if epoch > max_epoch:
                break;

        print("training done")

        # evaluate
        print("accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

        # save results
        minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ["output"])

        tf.train.write_graph(minimal_graph, 'result', 'graph.proto', as_text=False)
        tf.train.write_graph(minimal_graph, 'result', 'graph.txt', as_text=True)


if __name__ == '__main__':
    main()
