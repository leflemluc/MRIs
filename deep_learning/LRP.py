import tensorflow as tf
import numpy as np
import time
from utils.pickles import _read_pickle_, randomize
from utils.heatmap import visualize, heatmap
from deep_learning.simple_regression import SimpleRegression
from tensorflow.examples.tutorials.mnist import input_data
from utils.get_image_mnist import gen_image


mnist = input_data.read_data_sets("data/", one_hot=True)


#load MNIST dataset
#nacc = input_data.read_data_sets()
#INPUT_SIZE = 3 * 256 * 256
#OUTPUT_SIZE = 5

INPUT_SIZE = 784
OUTPUT_SIZE = 10

TRAINING_SIZE = 200
VALIDATION_SIZE = 20


checkpoint_dir = "/tmp/test_mris_2Dslices/model.ckpt"

if __name__ == '__main__':

    start_time = time.time()

    learning_rate = 0.1
    training_epochs = 10
    batch_size = 1
    display_step = 2


    # read
    # https://www.tensorflow.org/api_docs/python/tf/Graph
    with tf.Graph().as_default():

        # first build the structure of our neural network

        # variables has to be set up as placeholder before importing data
        x = tf.placeholder("float", [1, INPUT_SIZE])  # Data image: 3 slices of 256 * 256

        # y is the label in one-hot-encoding format
        y = tf.placeholder("float", [1, OUTPUT_SIZE])  # 5 stages detection

        # output is a matrix of probabilities

        n_layers = 3
        n_hidden_1 = 100
        n_hidden_2 = 200

        activations = [tf.nn.sigmoid, tf.nn.sigmoid, tf.identity]
        names = ["hidden_layer_1", "hidden_layer_2", "output_layer"]

        weight_shapes = [[INPUT_SIZE, n_hidden_1], [n_hidden_1, n_hidden_2], [n_hidden_2, OUTPUT_SIZE]]

        model = SimpleRegression(n_layers, activations, names, weight_shapes)

        #output = model.inference(x)

        relprop = model.relprop_RELU_activations(x, y)

        # define a session
        #init_op = tf.global_variables_initializer()

        # https://www.tensorflow.org/api_docs/python/tf/train/Saver
        saver = tf.train.Saver()


        sess = tf.Session()

        #sess.run(init_op)
        saver.restore(sess, checkpoint_dir)

        minibatch_x, minibatch_y = mnist.train.next_batch(batch_size)

        print("This is the initial image ")

        gen_image(minibatch_x[0]).show()
        rlp = sess.run(relprop, feed_dict={x: minibatch_x, y: minibatch_y})

        print("You can find the heatmap at /Users/lucleflem/Desktop/Ali/images/heatmap_third.png")
        visualize(rlp, heatmap, '/Users/lucleflem/Desktop/Ali/images/heatmap_third.png')
