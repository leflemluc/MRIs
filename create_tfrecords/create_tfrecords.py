import os
import time 
import pandas as pd 
import tensorflow as tf 
import numpy as np
import nibabel as nib
import sys 
from joblib import Parallel, delayed
import multiprocessing

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def normalize_n_create_tfrecords(file_names, minbycube, maxbycube, eps = 0.0001):
    count_fail=0
    examples = []
    #writer = tf.io.TFRecordWriter(save_filename)
    for index, row in file_names.iterrows():
        try:
            filename=row['filename']
            label=int(row['label'])
            name=str.encode(row['ind'])
            
            img = nib.load('data/'+filename).get_fdata()
            normalized_im=(img-minbycube)/(maxbycube+eps-minbycube)
            normalized_im = normalized_im.ravel()
            
            
            feature = {'label': _int64_feature(label),
                       'image': _float_feature(normalized_im),
                        'name': _bytes_feature(name)}
            
            # Create an example protocol buffer - Protocol buffers are a cross-platform, 
            #    cross-language library for efficient serialization of structured data.
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            examples.append(example)
            # Serialize to string and write on the file
            #writer.write(example.SerializeToString())
        except Exception as e: 
            print(e)
            count_fail+=1
    #writer.close()
    #return index, count_fail
    return examples

def write_tfrecords(save_filename, examples_per_processor):
    writer = tf.io.TFRecordWriter(save_filename)
    index = 0
    nb_written=0
    for examples in examples_per_processor:
        print("Writing examples of processor number " + str(index))
        index=1
        for example in examples: 
            writer.write(example.SerializeToString())
            nb_written+=1
    writer.close()
    return nb_written

def write_tfrecords_sequentially(save_filename, file_names, minbycube, maxbycube, eps = 0.0001):
    count_fail=0
    start_50 = time.time()
    writer = tf.io.TFRecordWriter(save_filename)
    for index, row in file_names.iterrows():
        if (index+1)%50 == 0:
            print(index)
            print("50 files in " + str(time.time() - start_50))
            start_50 = time.time()
        try:
            filename=row['filename']
            label=int(row['label'])
            name=str.encode(row['ind'])
            
            img = nib.load('data/'+filename).get_fdata()
            normalized_im=(img-minbycube)/(maxbycube+eps-minbycube)
            normalized_im = normalized_im.ravel()
            
            
            feature = {'label': _int64_feature(label),
                       'image': _float_feature(normalized_im),
                        'name': _bytes_feature(name)}
            
            # Create an example protocol buffer - Protocol buffers are a cross-platform, 
            #    cross-language library for efficient serialization of structured data.
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except Exception as e: 
            print(e)
            count_fail+=1
    writer.close()
    return index, count_fail
    
if __name__=="__main__":
    print("Starting")
    
    data_fold = sys.argv[1]
    train_or_test = sys.argv[2]
    print("Taking care of writing down " + train_or_test + " set " + str(data_fold))
    
    dataset = pd.read_csv('./datafolds/datafold_'+str(data_fold)+'/'+train_or_test+'_set.csv')
         
    print("Loading min and max from:")
    if train_or_test=="train":
        numpy_name = "TRAIN"
    else:
        numpy_name = "TEST"
    print('./datafolds/datafold_'+str(data_fold)+'/'+numpy_name+"_max_by_cube_matrix.npy")
    print("and")
    print('./datafolds/datafold_'+str(data_fold)+'/'+numpy_name+"_min_by_cube_matrix.npy")
    maxbycube = np.load('./datafolds/datafold_'+str(data_fold)+'/'+numpy_name+"_max_by_cube_matrix.npy")
    minbycube = np.load('./datafolds/datafold_'+str(data_fold)+'/'+numpy_name+"_min_by_cube_matrix.npy")
    
    print("Now its time to write the tf records: ")
    start = time.time()
    nb_written, fail = write_tfrecords_sequentially('./'+train_or_test+'_256_3d.tfrecords', dataset, minbycube, maxbycube)
    print("fails : " + str(fail))
    print("It took " + str(time.time() - start) + " to write the " + str(nb_written) + " tfrecords.")
    
    
    
    """
    print("Initializing parallelism")
    if train_or_test=="train":
        numpy_name = "TRAIN"
        numprocs = int(multiprocessing.cpu_count() * 3.5)

    else:
        numpy_name = "TEST"
        numprocs = multiprocessing.cpu_count()
    
    print("Num processors used : " + str(numprocs))
    
    parasize = int(len(dataset)/numprocs)
    slices = [(i*parasize, (i+1)*parasize) for i in range(numprocs)]
    slices[-1] = ((numprocs-1)*parasize, len(dataset)-1)
    
    print("Starting aggregating tf records")
    start = time.time()
    tfrecords_results = Parallel(n_jobs=numprocs)(delayed(normalize_n_create_tfrecords)(dataset.iloc[Slice[0]: Slice[1]], minbycube, maxbycube, eps = 0.0001) for Slice in slices)
    #print(tfrecords_results)
    index_proc = 0
    for examples in tfrecords_results:
        print("There are " + str(len(examples)) + " examples in proc " + str(index_proc))
    print("It took " + str(time.time() - start) + " to create all tfrecords")
    print("Now its time to write the tf records: ")
    start = time.time()
    nb_written = write_tfrecords('./'+train_or_test+'_256_3d.tfrecords', tfrecords_results)
    print("It took " + str(time.time() - start) + " to write the " + str(nb_written) + " tfrecords.")
    """

