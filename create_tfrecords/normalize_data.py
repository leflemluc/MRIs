import os
import time 
import pandas as pd 
import tensorflow as tf 
import numpy as np
import sys 
from joblib import Parallel, delayed
import multiprocessing

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

grid_cube_size = 8
n_cube_by_dim = 256 //grid_cube_size
n_of_cubes = (256 // 16)**3

def normalize(dataset):
    N = len(dataset)
    count_fail=0

    min_array = np.ones(256**3)*np.inf
    max_array = np.zeros(256**3)
    
    for index, row in dataset.iterrows():

        try:
            filename=row['filename']
            x=np.array(nib.load('./data/'+filename).get_fdata()).ravel()
            min_array = np.minimum(x, min_array)
            max_array = np.maximum(x, max_array)
            
        except Exception as e: 
            count_fail+=1
        
    return min_array, max_array, count_fail



if __name__ == "__main__":
    
    data_fold = sys.argv[1]
    train_or_test = sys.argv[2]
    dataset = pd.read_csv('./datafolds/datafold_'+str(data_fold)+'/'+train_or_test+'_set.csv')
    numprocs = multiprocessing.cpu_count() * 2
    
    print("Working on " + train_or_test + " dataset "+ str(data_fold))
    print("Initalizing multiprocessing")
    start = time.time()
    parasize = int(len(dataset)/numprocs)
    slices = [(i*parasize, (i+1)*parasize) for i in range(numprocs)]
    slices[-1] = ((numprocs-1)*parasize, len(dataset)-1)
                      
    start = time.time()
    print("Creating the max and min by pixel arrays for " + train_or_test)
    
    
    normlize_results = Parallel(n_jobs=numprocs)(delayed(normalize)(dataset.iloc[Slice[0]: Slice[1]]) for Slice in slices)
    max_x = np.maximum.reduce([normlize_results[i][1] for i in range(len(normlize_results))])
    min_x = np.minimum.reduce([normlize_results[i][0] for i in range(len(normlize_results))])
    
    #min_array, max_array, count_fail = normalize(dataset)
    
    print("Computing max and min vector in for " + train_or_test + " set " + str(data_fold)+' in '+ str(time.time() - start))
    #print("...with " + str(count_fail) + " fails")
    
    
    print("creating the max and min by cube arrays for " + train_or_test + " set " + str(data_fold))
    start = time.time()
    max_x = np.reshape(max_x, (256,256,256))
    min_x = np.reshape(min_x, (256,256,256))
        
    minbycube = np.ones([n_cube_by_dim, n_cube_by_dim, n_cube_by_dim])*np.inf
    maxbycube = np.zeros([n_cube_by_dim, n_cube_by_dim, n_cube_by_dim])
        
    for i in range(n_cube_by_dim):
        for j in range(n_cube_by_dim):
            for k in range(n_cube_by_dim):
                minbycube[i,j,k] = np.minimum(minbycube[i,j,k], np.min(min_x[grid_cube_size*i:grid_cube_size*(i+1), grid_cube_size*j:grid_cube_size*(j+1), grid_cube_size*k:grid_cube_size*(k+1)]))
                maxbycube[i,j,k] = np.maximum(maxbycube[i,j,k], np.max(max_x[grid_cube_size*i:grid_cube_size*(i+1), grid_cube_size*j:grid_cube_size*(j+1), grid_cube_size*k:grid_cube_size*(k+1)]))

    maxbycube_matrix = np.ones_like(max_x)
    minbycube_matrix = np.ones_like(min_x)
    for i in range(n_cube_by_dim):
        for j in range(n_cube_by_dim):
            for k in range(n_cube_by_dim):
                maxbycube_matrix[grid_cube_size*i:grid_cube_size*(i+1), grid_cube_size*j:grid_cube_size*(j+1), grid_cube_size*k:grid_cube_size*(k+1)] *= maxbycube[i, j, k]
                minbycube_matrix[grid_cube_size*i:grid_cube_size*(i+1), grid_cube_size*j:grid_cube_size*(j+1), grid_cube_size*k:grid_cube_size*(k+1)] *= minbycube[i, j, k]
                    
    print('...in '+ str(time.time() - start))
                    
    print("write min and max in a numpy file")
    if train_or_test == "test":
        NAME = "TEST"
    else:
        NAME = "TRAIN"
    np.save('./datafolds/datafold_'+str(data_fold)+'/' + NAME+'_min_by_cube_matrix', minbycube_matrix)
    np.save('./datafolds/datafold_'+str(data_fold)+'/' + NAME+'_max_by_cube_matrix', maxbycube_matrix)
