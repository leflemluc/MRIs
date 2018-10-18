import tensorflow as tf
from utils.layer_batch_normalization import layer_batch_normalization
from utils.image_summary import image_summary

# input size

input_size = 784
# re-constructed size
output_size = 784

"""
weight_shapes = [[INPUT_SIZE, 1000], [1000, 500], [500, 250], [250, N_OUT]]
"""

class Vanilla_autoencoders:
    def __init__(self, n_layers_coder, activations, names, weight_shapes):
        self.n_layers_coder = n_layers_coder #doesn't account for output layer
        self.names = names
        self.activations = activations
        self.weight_shapes = weight_shapes
        self.layers = []
        with tf.variable_scope("encoder"):
            for i in range(self.n_layers_coder - 1):
                with tf.variable_scope(names[i]):
                    #w_std = (2.0 / weight_shapes[i][0]) ** 0.5
                    w_std = 0.5

                    # initialization of the weights
                    # you can try either
                    #w_0 = tf.random_normal_initializer(stddev=w_std)
                    w_0 = tf.random_uniform_initializer(minval=-1,maxval=1)

                    b_0 = tf.constant_initializer(value=0)

                    W = tf.get_variable("W", weight_shapes[i], initializer=w_0)
                    b = tf.get_variable("b", weight_shapes[i][1], initializer=b_0)
                    self.layers.append([W, b])
            with tf.variable_scope("CODE"):
                # w_std = (2.0 / weight_shapes[i][0]) ** 0.5
                w_std = 0.5

                # initialization of the weights
                # you can try either
                # w_0 = tf.random_normal_initializer(stddev=w_std)
                w_0 = tf.random_uniform_initializer(minval=-1, maxval=1)
                b_0 = tf.constant_initializer(value=0)
                W = tf.get_variable("W", weight_shapes[-1], initializer=w_0)
                b = tf.get_variable("b", weight_shapes[-1][1], initializer=b_0)
                self.layers.append([W, b])

        with tf.variable_scope("decoder"):
            for i in range(self.n_layers_coder - 1):
                with tf.variable_scope(names[self.n_layers_coder - i - 1 - 1]):
                    # w_std = (2.0 / weight_shapes[i][0]) ** 0.5
                    w_std = 0.5

                    # initialization of the weights
                    # you can try either
                    # w_0 = tf.random_normal_initializer(stddev=w_std)
                    w_0 = tf.random_uniform_initializer(minval=-1, maxval=1)
                    b_0 = tf.constant_initializer(value=0)
                    W = tf.get_variable("W", [weight_shapes[self.n_layers_coder - i - 1][1],
                                              weight_shapes[self.n_layers_coder - i - 1][0]], initializer=w_0)
                    b = tf.get_variable("b", weight_shapes[self.n_layers_coder - i - 1][0], initializer=b_0)
                    self.layers.append([W, b])
            with tf.variable_scope("OUTPUT"):
                # w_std = (2.0 / weight_shapes[i][0]) ** 0.5
                w_std = 0.5
                # initialization of the weights
                # you can try either
                # w_0 = tf.random_normal_initializer(stddev=w_std)
                w_0 = tf.random_uniform_initializer(minval=-1, maxval=1)
                b_0 = tf.constant_initializer(value=0)
                W = tf.get_variable("W", [weight_shapes[0][1], weight_shapes[0][0]], initializer=w_0)
                b = tf.get_variable("b", weight_shapes[0][0], initializer=b_0)
                self.layers.append([W, b])

    def compute_layer_i(self, x, i, activation, phase_train, encode):
        if encode:
            weight_shape = self.weight_shapes[i][1]
        else:
            weight_shape = self.weight_shapes[self.n_layers_coder - i - 1][0]
            i = i + self.n_layers_coder
        logits = tf.matmul(x, self.layers[i][0]) + self.layers[i][1]
        return activation(layer_batch_normalization(logits, weight_shape, phase_train))


    def encoder(self, x, phase_train):
        for i in range(self.n_layers_coder):
            with tf.variable_scope("encoding_phase_number_" + str(i)):
                x = self.compute_layer_i(x, i, self.activations[i], phase_train, True)
        return x

    def decoder(self, x, phase_train):
        for i in range(self.n_layers_coder):
            with tf.variable_scope("decoding_phase_number_" + str(i)):
                x = self.compute_layer_i(x, i, self.activations[self.n_layers_coder - i - 1], phase_train, False)
        return x


    def loss(self, output, x):
        """
        Compute the loss of the auto-encoder
        intput:
            - output: the output of the decoder
            - x: true value of the sample batch - this is the input of the encoder

            the two have the same shape (batch_size * num_of_classes)
        output:
            - loss: loss of the corresponding batch (scalar tensor)

        """
        with tf.variable_scope("training"):
            l2_measure = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x)), 1))
            train_loss = tf.reduce_mean(l2_measure)
            return train_loss

    def evaluate(self, output, x, x_tilde):
        """
        evaluates the accuracy on the validation set
        input:
            -output: prediction vector of the network for the validation set
            -x: true value for the validation set
            -x_tilde: corrupted image with a level c
        output:
            - val_loss: loss of the autoencoder
            - in_image_op: input image
            - out_image_op:reconstructed image
            - val_summary_op: summary of the loss
        """

        with tf.variable_scope("validation"):
            # input of the autoencoder is the corrupted image
            in_image_op = image_summary("input_image", x_tilde, [28, 28])
            # output image
            out_image_op = image_summary("output_image", output, [28, 28])
            # the validation loss is computed using the original image
            l2_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(output, x, name="val_diff")), 1))

            val_loss = tf.reduce_mean(l2_norm)

            val_summary_op = tf.summary.scalar("val_cost", val_loss)

            return val_loss, in_image_op, out_image_op, val_summary_op



