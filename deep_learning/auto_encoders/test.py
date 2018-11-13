import argparse
import sys
from deep_learning.auto_encoders.vanilla_autoencoders import Vanilla_autoencoders
from deep_learning.auto_encoders.training import checkpoint_dir, INPUT_SIZE, batch_size, n_layers_coder, activations, names, weight_shapes
import tensorflow as tf
from utils.image_corruption import corrupt_1
import tensorflow.examples.tutorials.mnist.input_data as input_data

log_files_path = '/Users/lucleflem/Desktop/Ali/logs/autoencoder_test/'

if __name__ == '__main__':
    mnist = input_data.read_data_sets("data/", one_hot=True)

    with tf.Graph().as_default() as graph:

        with tf.variable_scope("autoencoder_model"):

            # the input variables are first define as placeholder
            # a placeholder is a variable/data which will be assigned later
            # image vector & label, phase_train is a boolean
            # MNIST data image of shape 28*28=784
            x = tf.placeholder("float", [None, INPUT_SIZE])

            phase_train = tf.placeholder(tf.bool)

            # ---------------------------------------------
            # corrupting (noising) data
            # the parameter c is also defined as a placeholder
            c = tf.placeholder(tf.float32)
            # x_tilde = x*(1.0 - c) + corrupt_x(x)*c
            x_tilde = corrupt_1(x)

            model = Vanilla_autoencoders(n_layers_coder, activations, names, weight_shapes)

            # define the encoder
            code = model.encoder(x_tilde, phase_train)

            # define the decoder
            output = model.decoder(code, phase_train)

            # compute the loss
            cost = model.loss(output, x)

            eval_op, in_image_op, out_image_op, val_summary_op = model.evaluate(
                output, x, x_tilde)
    with tf.Session(graph=graph) as sess:
        # Restore the model
        tf_saver = tf.train.Saver()
        tf_saver.restore(sess, checkpoint_dir)

        test_writer = tf.summary.FileWriter(log_files_path, graph=sess.graph)


        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            minibatch_x, minibatch_y = mnist.train.next_batch(
                batch_size)

            # Fit training using batch data
            # the training is done using the training dataset
            new_cost = sess.run(cost, feed_dict={
                x: minibatch_x, phase_train: False, c: 1.0})

            avg_cost += new_cost / total_batch

        print("Restoration of the previous session. Train loss =", "{:0.7f}".format(avg_cost))

        validation_loss, in_image, out_image, val_summary = sess.run(
            [eval_op, in_image_op, out_image_op, val_summary_op], feed_dict={
                x: mnist.validation.images, phase_train: False, c: 1.0})

        print("Validation Loss:", validation_loss)



        test_loss, in_image, out_image, val_summary = sess.run(
            [eval_op, in_image_op, out_image_op, val_summary_op], feed_dict={
                x: mnist.test.images, phase_train: False, c: 1.0})

        print("Test Loss:", test_loss)

        test_writer.add_summary(in_image, 0)
        test_writer.add_summary(out_image, 0)
        test_writer.add_summary(val_summary, 0)