import argparse
import sys
import numpy as np
import time
from utils.pickles import _read_pickle_, randomize
from deep_learning.simple_regression import SimpleRegression
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

#from process_data.dataset import Dataset
from process_data.dataset_V2 import Dataset



def main(_):
    print("Creating the dataset")

    """
    INPUT_SIZE = 256 * 256 * 256
    OUTPUT_SIZE = 5
    TRAINING_SIZE = 10
    VALIDATION_SIZE = 5
    TEST_SIZE = 1

    dataset = Dataset(TRAINING_SIZE, VALIDATION_SIZE, TEST_SIZE, True)

    """
    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    mnist = input_data.read_data_sets("data/", one_hot=True)

    start_time = time.time()

    learning_rate = 0.05
    training_epochs = 10
    batch_size = 100
    display_step = 1

    log_files_path = '/Users/lucleflem/Desktop/Ali/logs/test_mris_2Dslices/'
    checkpoint_dir = "/tmp/test_mris_2Dslices/model.ckpt"

    #log_files_path = '/home1/05990/tg852897/logs/test_mris_2Dslices/'
    #checkpoint_dir = "/home1/05990/tg852897/tmp/test_mris_2Dslices/",

    with tf.Graph().as_default() as graph:

        # first build the structure of our neural network

        # variables has to be set up as placeholder before importing data
        x = tf.placeholder("float", [None, INPUT_SIZE])  # Data image: 3 slices of 256 * 256

        # y is the label in one-hot-encoding format
        y = tf.placeholder("float", [None, OUTPUT_SIZE])  # 5 stages detection

        # output is a matrix of probabilities

        n_layers = 3
        n_hidden_1 = 100
        n_hidden_2 = 200

        # activations = [tf.nn.relu, tf.nn.relu, tf.identity]
        activations = [tf.nn.sigmoid, tf.nn.sigmoid, tf.identity]
        names = ["hidden_layer_1", "hidden_layer_2", "output_layer"]

        weight_shapes = [[INPUT_SIZE, n_hidden_1], [n_hidden_1, n_hidden_2], [n_hidden_2, OUTPUT_SIZE]]

        model = SimpleRegression(n_layers, activations, names, weight_shapes)

        output = model.inference(x)

        cost = model.loss(output, y)

        # set the initial value of global_step as 0
        # this will increase by 1 every time weights are updated
        global_step = tf.Variable(0, name='global_step', trainable=False)

        tf.summary.scalar("cost", cost)

        all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for variable in all_variables:
            tf.summary.histogram(variable.name, variable)

        # learning_rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, global_step=global_step)

        eval_op = model.evaluate(output, y)
        # eval_op = evaluate(output, y)

        summary_op = tf.summary.merge_all()

        # all variables need to be initialized by sess.run(tf.global_variables_initializer())
        init_op = tf.global_variables_initializer()

        # https://www.tensorflow.org/api_docs/python/tf/train/Saver
        saver = tf.train.Saver()

    # define a session
    with tf.Session(graph=graph) as sess:

        summary_writer = tf.summary.FileWriter(log_files_path, sess.graph)

        sess.run(init_op)

        loss_trace = []
        for epoch in range(training_epochs):
            avg_cost = 0.
            start_epoch = time.time()
            print("Starting epoch. Number of batches:")

            total_batch = int(mnist.train.num_examples / batch_size)
            #total_batch = int(dataset.trainset_size / batch_size)

            for i in range(total_batch):
                minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)
                #minibatch_x, minibatch_y = dataset.train_next_batch(batch_size)
                sess.run(train_op, feed_dict={x: minibatch_x, y: minibatch_y})
                # Compute average loss of all batches
                avg_cost += sess.run(cost, feed_dict={x: minibatch_x, y: minibatch_y}) / total_batch

            print("Epoch took " + str(time.time() - start_epoch))
            # Display logs per epoch step
            if epoch % display_step == 0:
                accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
                #accuracy = sess.run(eval_op, feed_dict={x: dataset.validation_inputs, y: dataset.validation_labels})
                loss_trace.append(1 - accuracy)
                print(
                    "Epoch:", '%03d' % (epoch + 1), "cost function=", "{:0.7f}".format(avg_cost), " Validation Error:",
                    (1.0 - accuracy))
                summary_str = sess.run(summary_op, feed_dict={x: minibatch_x, y: minibatch_y})
                summary_writer.add_summary(summary_str, sess.run(global_step))

                save_path = saver.save(sess, checkpoint_dir + "model.ckpt")

                #print("Model saved in path: %s" % save_path)

        print("Optimization Finished!")

        accuracy = sess.run(eval_op, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        #accuracy = sess.run(eval_op, feed_dict={x: dataset.tests_inputs, y: dataset.tests_labels})
        print("Test Accuracy:", accuracy)

        elapsed_time = time.time() - start_time

        print('Execution time was %0.3f' % elapsed_time)

        print("You can watch the tensorboard at ")
        print("tensorboard --logdir=" + str(log_files_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Your other flags go here.
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
