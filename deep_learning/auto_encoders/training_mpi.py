from deep_learning.auto_encoders.vanilla_autoencoders import Vanilla_autoencoders
import tensorflow as tf
from utils.image_corruption import corrupt_1
import tensorflow.examples.tutorials.mnist.input_data as input_data
import horovod.tensorflow as hvd

layers = tf.contrib.layers
learn = tf.contrib.learn

tf.logging.set_verbosity(tf.logging.INFO)


INPUT_SIZE = 784
OUTPUT_SIZE = 10

def main(_):
    learning_rate = 0.1
    training_epochs = 2
    batch_size = 64
    display_step = 1

    INPUT_SIZE = 784

    """
    # 3 hidden layers for encoder
    n_encoder_h_1 = 1000
    n_encoder_h_2 = 500
    n_encoder_h_3 = 250

    n_code = 10

    # 3 hidden layers for decoder
    n_decoder_h_1 = 250
    n_decoder_h_2 = 500
    n_decoder_h_3 = 1000
    """

    # 3 hidden layers for encoder
    n_encoder_h_1 = 10
    n_encoder_h_2 = 5
    n_encoder_h_3 = 2

    n_code = 1

    display_step = 5
    hvd.init()

    mnist = input_data.read_data_sets('data/MNIST-data-%d' % hvd.rank(), one_hot=True)

    with tf.Graph().as_default() as graph:

        with tf.variable_scope("autoencoder_model"):

            with tf.name_scope('input'):
                x = tf.placeholder("float", [None, INPUT_SIZE], name="image")

            phase_train = tf.placeholder(tf.bool)

            # ---------------------------------------------
            # corrupting (noising) data
            # the parameter c is also defined as a placeholder
            c = tf.placeholder(tf.float32)
            # x_tilde = x*(1.0 - c) + corrupt_x(x)*c
            x_tilde = corrupt_1(x)

            n_layers_coder = 4

            activations = [tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid]

            names = ["deep_1", "deep_2", "deep_3"]

            weight_shapes = [[INPUT_SIZE, n_encoder_h_1], [n_encoder_h_1, n_encoder_h_2],
                             [n_encoder_h_2, n_encoder_h_3], [n_encoder_h_3, n_code]]

            model = Vanilla_autoencoders(n_layers_coder, activations, names, weight_shapes)

            code = model.encoder(x_tilde, phase_train)

            # define the decoder
            output = model.decoder(code, phase_train)

            # compute the loss
            cost = model.loss(output, x)

            tf.summary.scalar("cost", cost)
            all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for variable in all_variables:
                tf.summary.histogram(variable.name, variable)

            opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())

            opt = hvd.DistributedOptimizer(opt)

            global_step = tf.contrib.framework.get_or_create_global_step()
            train_op = opt.minimize(cost, global_step=global_step)
            eval_op, in_image_op, out_image_op, val_summary_op = model.evaluate(
                output, x, x_tilde)

            log_files_path = './summary_dir' if hvd.rank() == 0 else None
            summary_op = tf.summary.merge_all()
            hooks = [
                # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
                # from rank 0 to all other processes. This is necessary to ensure consistent
                # initialization of all workers when training is started with random weights
                # or restored from a checkpoint.
                hvd.BroadcastGlobalVariablesHook(0),

                # Horovod: adjust number of steps based on number of GPUs.
                tf.train.StopAtStepHook(last_step=100),

                tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': cost},
                                           every_n_iter=10),

                tf.train.SummarySaverHook(save_steps=5, output_dir=log_files_path, summary_op=summary_op)
            ]


        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

        # Horovod: save checkpoints only on worker 0 to prevent other workers from
        # corrupting them.
        checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                               hooks=hooks,
                                               config=config) as sess:
            loss_trace = []
            while not sess.should_stop():
                # Run a training step synchronously.
                minibatch_x, minibatch_y = mnist.train.next_batch(
                    batch_size)

                # Fit training using batch data
                # the training is done using the training dataset
                _, new_cost = sess.run([train_op, cost], feed_dict={
                    x: minibatch_x, phase_train: True, c: 1.0})


if __name__ == "__main__":
        tf.app.run()
