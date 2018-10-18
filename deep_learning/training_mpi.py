import tensorflow as tf
import horovod.tensorflow as hvd
from deep_learning.simple_regression import SimpleRegression
import tensorflow.examples.tutorials.mnist.input_data as input_data

layers = tf.contrib.layers
learn = tf.contrib.learn

tf.logging.set_verbosity(tf.logging.INFO)


INPUT_SIZE = 784
OUTPUT_SIZE = 10

def main(_):

        display_step = 5
        # log_files_path = './summary_dir' if hvd.rank() == 0 else None
        # Horovod: initialize Horovod.
        hvd.init()

        # Download and load MNIST dataset.

        mnist = input_data.read_data_sets('data/MNIST-data-%d' % hvd.rank(), one_hot=True)

        # Build model...
        with tf.name_scope('input'):
            image = tf.placeholder(tf.float32, [None, INPUT_SIZE], name='image')
            label = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='label')

        n_layers = 3
        n_hidden_1 = 100
        n_hidden_2 = 200

        # activations = [tf.nn.relu, tf.nn.relu, tf.identity]
        activations = [tf.nn.sigmoid, tf.nn.sigmoid, tf.identity]
        names = ["hidden_layer_1", "hidden_layer_2", "output_layer"]

        weight_shapes = [[INPUT_SIZE, n_hidden_1], [n_hidden_1, n_hidden_2], [n_hidden_2, OUTPUT_SIZE]]

        model = SimpleRegression(n_layers, activations, names, weight_shapes)

        output = model.inference(image)

        loss = model.loss(output, label)

        tf.summary.scalar("cost", loss)
        all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for variable in all_variables:
            tf.summary.histogram(variable.name, variable)
        # Horovod: adjust learning rate based on number of GPUs.
        opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())

        # Horovod: add Horovod Distributed Optimizer.
        opt = hvd.DistributedOptimizer(opt)

        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = opt.minimize(loss, global_step=global_step)
        eval_op = model.evaluate(output, y)

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

            tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
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
                image_, label_ = mnist.train.next_batch(100)
                _, step = sess.run([train_op, global_step], feed_dict={image: image_, label: label_})

                if hvd.rank() == 0 and step%display_step==0:
                    accuracy = sess.run(eval_op, feed_dict={image: mnist.validation.images, label: mnist.validation.labels})
                    # accuracy = sess.run(eval_op, feed_dict={x: dataset.validation_inputs, y: dataset.validation_labels})
                    loss_trace.append(1 - accuracy)
                    print(
                        "Epoch:", '%03d' % (step + 1),
                        " Validation Error:",
                        (1.0 - accuracy))


if __name__ == "__main__":
        tf.app.run()
