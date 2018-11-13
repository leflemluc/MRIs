import tensorflow as tf


def image_summary(label, tensor, shape_2D):
    # tf.summary.image: https://www.tensorflow.org/api_docs/python/tf/summary/image
    # Outputs a Summary protocol buffer with images.

    tensor_reshaped = tf.reshape(tensor, [-1, shape_2D[0], shape_2D[1], 1])
    return tf.summary.image(label, tensor_reshaped, max_outputs=3)
