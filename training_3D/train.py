import sys 
import os 
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
from utils import _parse_function, _select_patch_and_reshape, MRI_SIZE, MODIFIED_SIZE, NUM_CHANNEL, OUTPUT_SIZE
from model import inference, loss

logdir = 'logs_3D_CNN_full_156_patches_withL2loss_2_LR_1_B256/'
chkpt = 'logs_3D_CNN_full_156_patches_withL2loss_2_LR_1_B256/model.ckpt'


KEEP_RATE = 0.5

PATH_TO_DATA = './Create_TFrecords/datafolds/'

class Trainer:
    

    def __init__(self, datafold, adam_rate=0.0001, batch_size = 256, n_epochs = 30, penalty_intensity = 0.05):
        
        path_folder = PATH_TO_DATA+ 'datafold_' + str(datafold) + '/'
        
        train_csv = pd.read_csv(path_folder+"train_set.csv")
        self.training_set_size = len(train_csv)
        self.train_tf_records_path = path_folder+'train_256_3d.tfrecords'
        
        test_csv = pd.read_csv(path_folder+"test_set.csv")
        self.test_set_size = len(test_csv)
        self.test_tf_records_path = path_folder+'test_256_3d.tfrecords'
      
        self.adam_rate = adam_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.penalty_intensity = penalty_intensity
        print("adam_rate: " + str(adam_rate))
        print("batch_size: " + str(batch_size))
        print("n_epochs: " + str(n_epochs))
        print("penalty_intensity: " + str(penalty_intensity))
        
        self.logdir = path_folder + '/logs_3D_CNN_LR_'+str(adam_rate)+'_BS_'+str(batch_size)+'_L2_'+ str(penalty_intensity) + '/'
        self.tensorboard_n_checkpoint = self.logdir + 'tensorboard_n_checkpoint/'
        self.chkpt = self.tensorboard_n_checkpoint + 'model.ckpt'

        
        with tf.variable_scope('3D_CNN'):
            
            self.X = tf.placeholder(tf.float32, [None, MODIFIED_SIZE, MODIFIED_SIZE, MODIFIED_SIZE, NUM_CHANNEL], name='X')
            self.y = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name='y')
            self.keep_rate = tf.placeholder(tf.float32)
            score  = inference(self.X, self.keep_rate, OUTPUT_SIZE)
            softmax = tf.nn.softmax(score)
            self.cost = loss(score, self.y, self.penalty_intensity)
                
            self.optimizer = tf.train.AdamOptimizer(self.adam_rate).minimize(self.cost, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
            
            self.preds = tf.equal(tf.argmax(softmax, axis=1), tf.argmax(self.y, axis=1))
            self.accuracy = tf.reduce_mean(tf.cast(self.preds, tf.float32))

        self.cost_summary = tf.summary.scalar(name='Cost', tensor=self.cost)
        
        self.accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=self.accuracy)

        self.summary = tf.summary.merge_all()

    def run(self):
        
        with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
            
            sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
            
            
            if os.path.exists(self.logdir):
                print("restoring pre-existant weights")
                self.saver.restore(sess, tf.train.latest_checkpoint(self.tensorboard_n_checkpoint)) 
            
            
            self.file_writer = tf.summary.FileWriter(self.tensorboard_n_checkpoint, tf.get_default_graph())
            
            self.filenames = tf.placeholder(tf.string, shape=[None])
            #self.dataset = tf.data.TFRecordDataset(self.filenames).map(_parse_function).shuffle(buffer_size=buffer_size_validation_set).batch(batch_size).repeat()

            self.dataset = tf.data.TFRecordDataset(self.filenames).map(_parse_function).batch(self.batch_size).repeat()
            self.iterator = self.dataset.make_initializable_iterator()
            self.next_element = self.iterator.get_next()
            self.training_filenames = [self.train_tf_records_path]
            sess.run(self.iterator.initializer, feed_dict={self.filenames: self.training_filenames})

            self.val_filenames = tf.placeholder(tf.string, shape=[None])
            #self.val_dataset = tf.data.TFRecordDataset(self.val_filenames).map(_parse_function).shuffle(buffer_size=buffer_size_test_set).batch(batch_size).repeat()
            self.val_dataset = tf.data.TFRecordDataset(self.val_filenames).map(_parse_function).batch(self.batch_size).repeat()
            self.val_iterator = self.val_dataset.make_initializable_iterator()
            self.val_next_element = self.val_iterator.get_next()
            self.validation_filenames = [self.test_tf_records_path]
            sess.run(self.val_iterator.initializer, feed_dict={self.val_filenames: self.validation_filenames})


            for epoch in range(n_epochs):
                self.train(sess, epoch)
                self.validate(sess)
                self.saver.save(sess, chkpt)
                        

    
    def train(self, sess, epoch):
        import time
        n_batches=self.training_set_size//self.batch_size
        avg_cost = 0
        avg_accuracy = 0
        times = 0
        time_10batches=0
        for batch in range(n_batches):
            t0 = time.time()
            x_batch, y_batch, _name = sess.run(self.next_element)
            #patch_x = _select_patch_and_reshape(x_batch, patch_size)
            _, batch_cost, batch_accuracy, summ = sess.run([self.optimizer, self.cost, self.accuracy, self.summary], feed_dict={self.X: x_batch, self.y: y_batch, self.keep_rate: KEEP_RATE})
            avg_cost += batch_cost
            avg_accuracy += batch_accuracy
            self.file_writer.add_summary(summ, epoch * n_batches + batch)
            completion = batch / n_batches
            print_str = '|' + int(completion * 20) * '#' + (19 - int(completion * 20))  * ' ' + '|'
            print('\rEpoch {0:>3} {1} {2:3.0f}% Cost {3:6.4f} Accuracy {4:6.4f}'.format('#' + str(epoch + 1), 
                print_str, completion * 100, avg_cost / (batch + 1), avg_accuracy / (batch + 1)), end='')
            t1 = time.time()
            batch_time = t1-t0
            times+=batch_time
            time_10batches+=batch_time
            
            if (batch+1)%10:
                print(" 10 batches took " + str(time_10batches))
                time_10batches = 0
            
        print(end=' ')
        print("Epoch took " + str(times))    


    def validate(self, sess):
        #TODO: access full size of dataset
        n_batches = self.test_set_size//self.batch_size
        avg_accuracy = 0
        for batch in range(n_batches):
            x_batch, y_batch, _name = sess.run(self.val_next_element)
            patch_x = _select_patch_and_reshape(x_batch, patch_size)
            avg_accuracy += sess.run([self.accuracy, ], feed_dict={self.X: patch_x, self.y: y_batch, self.keep_rate: 0.0})[0]

        avg_accuracy /= n_batches
        print('Validation Accuracy {0:6.4f}'.format(avg_accuracy))
    

if __name__ == '__main__':
    data_fold = sys.argv[1]
    adam_rate = float(sys.argv[2])
    batch_size = int(sys.argv[3])
    n_epochs = int(sys.argv[4])
    penalty_intensity = float(sys.argv[5])
    Trainer(data_fold, adam_rate, batch_size, n_epochs, penalty_intensity).run()
    #datafold, adam_rate=0.0001, batch_size = 256, n_epochs = 30, penalty_intensity = 0.05

