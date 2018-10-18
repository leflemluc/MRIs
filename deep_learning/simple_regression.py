import tensorflow as tf

class SimpleRegression:
    def __init__(self, n_layers, activations, names, weight_shapes):
        self.n_layers = n_layers #doesn't account for output layer
        self.names = names
        self.activations = activations
        self.weight_shapes = weight_shapes
        self.layers = []
        for i in range(self.n_layers):
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


    def compute_layer_i(self, x, i, activation):
        return activation(tf.matmul(x, self.layers[i][0]) + self.layers[i][1])

    def inference(self, x):
        for i in range(self.n_layers):
            x = self.compute_layer_i(x, i, self.activations[i])
        return tf.nn.softmax(x)

    def loss(self, output, y):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
        loss = tf.reduce_mean(xentropy)

        return loss

    def evaluate(self, output, y):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        # tf.cast transfer boolean tensor into float tensor
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("validation_error", (1.0 - accuracy))

        return accuracy



    def relprop_RELU_activations(self, input, label):
        layers = []
        for i in range(self.n_layers):
            layers.append(input)
            input = self.compute_layer_i(input, i, self.activations[i])
        R = input * tf.cast(label, tf.float32)
        i = self.n_layers
        for l in self.layers[::-1]:
            layer = layers[i-1]
            i -= 1
            V = tf.maximum(tf.cast(0, tf.float32), l[0])
            Z = tf.tensordot(layer, V, axes=1) + 1e-9
            S = R / Z
            C = tf.tensordot(S, tf.transpose(V), axes=1)
            R = layer * C

        #return tf.reshape(R, shape=(28, 28))
        return R






