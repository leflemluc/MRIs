import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
from utils import _parse_function
from matplotlib import pyplot as plt

import pandas as pd
testing_labels = pd.read_csv('data/testing_file_names.csv')

testing_labels[:3]

def get_ten_images():
    batch_size=10
    sess = tf.Session()
    filenames_x = tf.placeholder(tf.string, shape=[None])
    dataset_x = tf.data.TFRecordDataset(filenames_x).map(_parse_function).batch(batch_size)
    iterator_x = dataset_x.make_initializable_iterator()
    next_element_x = iterator_x.get_next()

    filenames_y = tf.placeholder(tf.string, shape=[None])
    dataset_y = tf.data.TFRecordDataset(filenames_y).map(_parse_function).batch(batch_size)
    iterator_y = dataset_y.make_initializable_iterator()
    next_element_y = iterator_y.get_next()

    filenames_z = tf.placeholder(tf.string, shape=[None])
    dataset_z = tf.data.TFRecordDataset(filenames_z).map(_parse_function).batch(batch_size)
    iterator_z = dataset_z.make_initializable_iterator()
    next_element_z = iterator_z.get_next()

    training_filenames_x = ["../testing_flat_x_156_full_dataset.tfrecords"]
    training_filenames_y = ["../testing_flat_y_156_full_dataset.tfrecords"]
    training_filenames_z = ["../testing_flat_z_156_full_dataset.tfrecords"]

    sess.run(iterator_x.initializer, feed_dict={filenames_x: training_filenames_x})
    sess.run(iterator_y.initializer, feed_dict={filenames_y: training_filenames_y})
    sess.run(iterator_z.initializer, feed_dict={filenames_z: training_filenames_z})
    images_x, labels_x, name_x = sess.run(next_element_x)
    images_y, labels_y, name_y = sess.run(next_element_y)
    images_z, labels_z, name_z = sess.run(next_element_z)
    
    return images_x, labels_x, name_x, images_y, labels_y, name_y, images_z, labels_z, name_z


def _parse_label(np_array):
    if np.argmax(np_array) == 0:
        label = 'CN'
    if np.argmax(np_array) == 1:
        label = 'AD'
    if np.argmax(np_array) == 2:
        label = 'EMCI'
    if np.argmax(np_array) == 3:
        label = 'CMCI'
    if np.argmax(np_array) == 4:
        label = 'MCI'
    return label


def create_images(N):
    for i in range(N):
        images_x, labels_x, name_x, images_y, labels_y, name_y, images_z, labels_z, name_z = get_ten_images()
        for j in range(10):
            print('New image below')
            label = _parse_label(labels_x[j])
            scale = 5
            fig, axes = plt.subplots(2,3)# squeeze=False)
            plt.axis('off')
            fig.set_size_inches(5*scale,5*scale)
            fig.subplots_adjust(hspace=-0.5)
            plt.tight_layout()
            MRI_X = images_x[j].reshape([156, 156])
            axes[0][0].imshow(MRI_X, cmap='gray')
            axes[0][0].set_title(label)
            MRI_Y = images_y[j].reshape([156, 156])
            axes[0][1].imshow(MRI_Y, cmap='gray')
            axes[0][1].set_title(label)
            MRI_Z = images_z[j].reshape([156, 156])
            axes[0][2].imshow(MRI_Z, cmap='gray')
            axes[0][2].set_title(label)
            
            lrp_x = mpimg.imread('results_x_neg/'+name_x[j].decode()+'_x.png')
            axes[1][0].imshow(lrp_x)
            axes[1][0].axis('off')
            axes[1][0].set_title(label)
            
            lrp_y = mpimg.imread('results_y_neg/'+name_x[j].decode()+'_y.png')
            axes[1][1].imshow(lrp_y)
            axes[1][1].axis('off')
            axes[1][1].set_title(label)
            
            lrp_z = mpimg.imread('results_z_neg/'+name_x[j].decode()+'_z.png')
            axes[1][2].imshow(lrp_z)
            axes[1][2].axis('off')
            axes[1][2].set_title(label)
            title = name_x[j].decode() + '  :  ' + label
            fig.suptitle(title)
            plt.show()
            print('\n\n\n')


create_images(5) # produces 10 times number of images
