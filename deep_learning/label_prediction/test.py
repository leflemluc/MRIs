import argparse
import sys
import time
from deep_learning.label_prediction.simple_regression import SimpleRegression
from deep_learning.label_prediction.training import batch_size, checkpoint_dir, n_layers, activations, names, weight_shapes

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

#from process_data.dataset import Dataset


def main(_):

    from_mpi = sys.argv[1]

    if from_mpi:
        restore_dir = "./checkpoints"
    else:
        restore_dir = checkpoint_dir

    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    mnist = input_data.read_data_sets("data/", one_hot=True)
    total_batch = int(mnist.train.num_examples / batch_size)

    with tf.Graph().as_default() as graph:

        # first build the structure of our neural network

        # variables has to be set up as placeholder before importing data
        x = tf.placeholder("float", [None, INPUT_SIZE])  # Data image: 3 slices of 256 * 256

        # y is the label in one-hot-encoding format
        y = tf.placeholder("float", [None, OUTPUT_SIZE])  # 5 stages detection

        # output is a matrix of probabilities

        # activations = [tf.nn.relu, tf.nn.relu, tf.identity]

        model = SimpleRegression(n_layers, activations, names, weight_shapes)

        output = model.inference(x)

        cost = model.loss(output, y)

        eval_op = model.evaluate(output, y)

    with tf.Session(graph=graph) as sess:
        # Restore the model
        tf_saver = tf.train.Saver()
        tf_saver.restore(sess, restore_dir)
        avg_cost = 0
        loss_trace= []
        for i in range(total_batch):
            minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
            avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y}) / total_batch
        accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
        # accuracy = sess.run(eval_op, feed_dict={x: dataset.validation_inputs, y: dataset.validation_labels})
        loss_trace.append(1 - accuracy)
        print(
            "Restoration of the previous session. Train loss =", "{:0.7f}".format(avg_cost), " Validation Error:",
            (1.0 - accuracy))

        accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        #accuracy = sess.run(eval_op, feed_dict={x: dataset.tests_inputs, y: dataset.tests_labels})
        print("Test Accuracy:", accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Your other flags go here.
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

