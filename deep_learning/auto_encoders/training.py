import argparse
import sys
from deep_learning.auto_encoders.vanilla_autoencoders import Vanilla_autoencoders
import tensorflow as tf
from utils.image_corruption import corrupt_1
import tensorflow.examples.tutorials.mnist.input_data as input_data


batch_size = 64
INPUT_SIZE = 784

# 3 hidden layers for encoder
n_encoder_h_1 = 10
n_encoder_h_2 = 5
n_encoder_h_3 = 2

n_code = 1


n_layers_coder = 4

activations = [tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid]

names = ["deep_1", "deep_2", "deep_3"]

weight_shapes = [[INPUT_SIZE, n_encoder_h_1], [n_encoder_h_1, n_encoder_h_2],
                             [n_encoder_h_2, n_encoder_h_3], [n_encoder_h_3, n_code]]


log_files_path = '/Users/lucleflem/Desktop/Ali/logs/autoencoders/'
checkpoint_dir = "/Users/lucleflem/Desktop/Ali/tmp/autoencoders/model.ckpt"

def main(_):

    mnist = input_data.read_data_sets("data/", one_hot=True)

    learning_rate = 0.1
    training_epochs = 2
    display_step = 1

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

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_summary_op = tf.summary.scalar("train_cost", cost)

            # learning_rate
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(cost, global_step=global_step)

            # evaluate the accuracy of the network (done on a validation set)
            eval_op, in_image_op, out_image_op, val_summary_op = model.evaluate(
                output, x, x_tilde)

            summary_op = tf.summary.merge_all()

        # all variables need to be initialized by sess.run(tf.global_variables_initializer())
        init_op = tf.global_variables_initializer()

        # https://www.tensorflow.org/api_docs/python/tf/train/Saver
        saver = tf.train.Saver()

    # define a session
    with tf.Session(graph=graph) as sess:

        train_writer = tf.summary.FileWriter(log_files_path, graph=sess.graph)
        val_writer = tf.summary.FileWriter(log_files_path, graph=sess.graph)

        sess.run(init_op)

        loss_trace = []
        for epoch in range(training_epochs):

            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)

            # Loop over all batches
            for i in range(total_batch):
                minibatch_x, minibatch_y = mnist.train.next_batch(
                    batch_size)

                # Fit training using batch data
                # the training is done using the training dataset
                _, new_cost, train_summary = sess.run([train_op, cost, train_summary_op], feed_dict={
                    x: minibatch_x, phase_train: True, c: 1.0})

                train_writer.add_summary(
                    train_summary, sess.run(global_step))

                # Compute average loss
                avg_cost += new_cost / total_batch

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost =", "{:.9f}".format(avg_cost))

                train_writer.add_summary(
                    train_summary, sess.run(global_step))

                validation_loss, in_image, out_image, val_summary = sess.run(
                        [eval_op, in_image_op, out_image_op, val_summary_op], feed_dict={
                            x: mnist.validation.images, phase_train: False, c: 1.0})

                val_writer.add_summary(in_image, sess.run(global_step))
                val_writer.add_summary(out_image, sess.run(global_step))
                val_writer.add_summary(val_summary, sess.run(global_step))

                print("Validation Loss:", validation_loss)

                save_path = saver.save(sess, checkpoint_dir)
                print("Model saved in file: %s" % save_path)

        print("Optimization Done")
        print("You can watch the tensorboard at ")
        print("tensorboard --logdir=" + str(log_files_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Your other flags go here.
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
