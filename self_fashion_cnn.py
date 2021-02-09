"""
Classification example: Iris data set
https://archive.ics.uci.edu/ml/datasets/Iris

Author: Joel Diebe (work based on Stefano Melacci scripts)

"""

import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.disable_v2_behavior()
tf.reset_default_graph()


def target_encoding(raw_target):
    """ One hot encode the target labels """

    rows = raw_target.shape[0]

    # converting class labels to 1-hot representations (targets)
    targets_1hot = np.zeros((raw_target.shape[0], np.max(raw_target) + 1))
    targets_1hot[np.arange(rows), raw_target] = 1

    return targets_1hot


def create_hidden_layer(input_data, filter_size, kernel_size, layer_id):
    A = tf.layers.conv2d(input_data, filter_size, kernel_size, activation=None, use_bias=True,
                         name="Conv2D_" + str(layer_id))
    return tf.nn.relu(A)


def create_pooling_layer(input_data, pool_size, stride_size):
    return tf.layers.max_pooling2d(input_data, pool_size, stride_size)


def create_flatten_layer(input_data):
    return tf.layers.flatten(input_data)


def create_output_layer(input_data, layer_size, layer_id):
    return tf.layers.dense(input_data, layer_size, activation=None, use_bias=True,
                           kernel_initializer=tf.random_normal_initializer(),
                           bias_initializer=tf.random_normal_initializer(),
                           name="Dense" + str(layer_id))


def create_loss_function(net_output, Y):
    loss_on_each_example = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=net_output)
    return tf.reduce_mean(loss_on_each_example)


def compute_gradient_update_weights(loss, learning_rate):
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


def compute_accuracy(net_output, Y):
    correct_predictions = tf.equal(tf.argmax(net_output, axis=1), tf.argmax(Y, axis=1))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) * 100.0


def create_network():
    d = 28  # number of input features
    c = 10  # number of classes

    X = tf.placeholder(tf.float32, [None, d, d, 1])
    Y = tf.placeholder(tf.float32, [None, c])

    layer_output = create_hidden_layer(X, 16, 3, 1)
    layer_output = create_pooling_layer(layer_output, 2, 1)
    layer_output = create_hidden_layer(layer_output, 16, 3, 2)
    layer_output = create_pooling_layer(layer_output, 2, 1)
    layer_output = create_hidden_layer(layer_output, 16, 3, 3)
    layer_output = create_flatten_layer(layer_output)
    net_output = create_output_layer(layer_output, c, 4)

    loss = create_loss_function(net_output, Y)

    learning_step = compute_gradient_update_weights(loss, 0.001)

    accuracy = compute_accuracy(net_output, Y)

    return X, Y, loss, learning_step, accuracy


# loading data set
_data = np.load("fashion_test_data.npy")
_targets = np.load("fashion_test_labels.npy")

_data_train, test_set_data, _targets_train, test_set_targets = train_test_split(_data, _targets,
                                                                                train_size=0.75,
                                                                                random_state=0,
                                                                                stratify=_targets)

train_set_data, val_set_data, train_set_targets, val_set_targets = train_test_split(_data_train, _targets_train,
                                                                                    train_size=0.66,
                                                                                    random_state=1,
                                                                                    stratify=_targets_train)


train_set_data = train_set_data.reshape((train_set_data.shape[0], 28, 28, 1)) / 255
val_set_data = val_set_data.reshape((val_set_data.shape[0], 28, 28, 1)) / 255
test_set_data = test_set_data.reshape((test_set_data.shape[0], 28, 28, 1)) / 255

train_set_targets = target_encoding(train_set_targets)
val_set_targets = target_encoding(val_set_targets)
test_set_targets = target_encoding(test_set_targets)


# creating the network and defining all the needed TensorFlow operations
_X, _Y, _loss, _learning_step, _accuracy = create_network()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # training epochs
    for i in range(0, 5000):
        # computing loss, updating weights, computing accuracy
        # on the training set data
        output = sess.run([_loss, _learning_step, _accuracy],
                          feed_dict={_X: train_set_data, _Y: train_set_targets})

        # computing accuracy on data belonging to the validation set
        val_acc = sess.run(_accuracy, feed_dict={_X: val_set_data, _Y: val_set_targets})

        print("Epoch " + str(i) +
              ", Loss=" + "{0:f}".format(output[0]) +
              ", TrainAccuracy=" + "{0:.2f}".format(output[2]) + "%" +
              ", ValAccuracy=" + "{0:.2f}".format(val_acc) + "%")

    # computing accuracy on data belonging to the test set
    test_acc = sess.run(_accuracy, feed_dict={_X: test_set_data, _Y: test_set_targets})

    print("TestAccuracy=" + "{0:.2f}".format(test_acc) + "%")
