import tensorflow as tf


def inference(x, keep_rate, classNum):
        with tf.name_scope("layer_a"):
            conv1 = tf.layers.conv3d(inputs=x, filters=16, kernel_size=[3,3,3],strides=(2,2,2), padding='same', activation=tf.nn.relu)
            print(conv1)
            conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=[3,3,3],strides=(2,2,2), padding='same', activation=tf.nn.relu)
            print(conv2)
            pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)
            cnn3d_bn_1 = tf.layers.batch_normalization(inputs=pool3, training=True)

        with tf.name_scope("layer_c"):
            # conv => 8*8*8
            conv4 = tf.layers.conv3d(inputs=cnn3d_bn_1, filters=64, kernel_size=[3,3,3],strides=(2,2,2), padding='same', activation=tf.nn.relu)
            print(conv4)
            conv5 = tf.layers.conv3d(inputs=conv4, filters=128, kernel_size=[3,3,3],strides=(2,2,2), padding='same', activation=tf.nn.relu)
            print(conv5)
            
            pool6 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[2, 2, 2], strides=2)
            print(pool6)
            cnn3d_bn_2 = tf.layers.batch_normalization(inputs=pool6, training=True)
            print(cnn3d_bn_2)
        with tf.name_scope("fully_con"):
            flattening = tf.reshape(cnn3d_bn_2, [-1, 2*2*2*128])
            print(flattening)
            dense = tf.layers.dense(inputs=flattening, units=1024, activation=tf.nn.relu)
            # (1-keep_rate) is the probability that the node will be kept
            dropout = tf.layers.dropout(inputs=dense, rate=keep_rate, training=True)
            print(dropout)

        with tf.name_scope("y_conv"):
            y_conv = tf.layers.dense(inputs=dropout, units=classNum)
            print(y_conv)
            
            
        return y_conv 
    
    
    
def loss(score, y, penalty_intensity=0.05):
    vars_ = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars_]) * penalty_intensity
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=y)) + lossL2

            
    



