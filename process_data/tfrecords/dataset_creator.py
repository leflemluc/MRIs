from tensorflow.train import Feature, FloatList, BytesList
# Define feature converter for flattened np.uint8 arrays
import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input_data

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(vals):
    return Feature(bytes_list=BytesList(value=[vals]))
# Define feature converter for flattened np.float32 arrays

def _floats_feature(vals):
    return Feature(float_list=FloatList(value=[float(x) for x in vals]))

def create_tfrecords(images, labels, path_to_save):
    writer = tf.python_io.TFRecordWriter(path_to_save)
    for i in range(len(images)):
        img = images[i]
        label = labels[i]

        feature = {'train/label': _floats_feature(label),
                   'train/image': _floats_feature(tf.compat.as_bytes(img.tostring()))}

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())
    writer.close()




if __name__ == "__main__":
    mnist = input_data.read_data_sets("data/", one_hot=True)
    images = mnist.train.images
    labels = mnist.train.labels
    train_filename = './train.tfrecords'
    create_tfrecords(images, labels, train_filename)
    train_dataset = tf.data.TFRecordDataset("train.tfrecords")
    train_dataset.shuffle(buffer_size=1000)
    training_ds = train_dataset.batch(64)

    handle = tf.placeholder(tf.string, shape=[], name="dh")
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    next_element = iterator.get_next()

    training_iterator = train_dataset.make_initializable_iterator()

    with tf.Session() as sess:
        training_handle = sess.run(training_iterator.string_handle())
        sess.run(training_iterator.initializer)
        count_training = 0
        while not sess.should_stop():
            image_, label_ = sess.run(next_element, feed_dict={handle: training_handle})


