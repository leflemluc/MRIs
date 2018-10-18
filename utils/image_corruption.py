import tensorflow as tf

def corrupt_1(x):
    # Take an input tensor and add uniform masking
    # Parameters
    # ----------
    # x : Tensor/Placeholder
    #    Input to corrupt.
    # Returns
    # -------
    # x_corrupted : Tensor
    # 50 pct of values corrupted.

    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32), tf.float32))


def corrupt_2(x):
    """
    Take an input tensor and add uniform masking.
    input:
        - x : input vector - to be corrupted
    output:
        - output vector : half of the values are corrupted
    """

    # https://www.tensorflow.org/api_docs/python/tf/random_uniform
    # Outputs random values from a uniform distribution.
    # The lower bound minval is included in the range, while the upper bound maxval is excluded.
    # here 0 or 1
    c = tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32)

    # https://www.tensorflow.org/api_docs/python/tf/cast
    # change the type to float

    return x * tf.cast(c, tf.float32)