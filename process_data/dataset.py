import pandas as pd
import numpy as np
from process_data.prepare_dataset import PATH_TOSAVE_DATA, PATH_DIAGNOSIS_PICKLE, PATH_MRI
from process_data.prepare_dataset import create_dataset, compute_inbalance


PATH_DATASET_TWOD = "/Users/lucleflem/Desktop/Ali/data/full_dataset.p"
#PATH_DATASET_TWOD = "/Users/lucleflem/Desktop/Ali/data/full_dataset_2D.p"
PATH_DATASET_THREED = "/Users/lucleflem/Desktop/Ali/data/full_dataset_3D.p"



class Dataset:
    def __init__(self, train_size, validation_size, randomize_whole_dataset, TwoD=True):

        if TwoD:
            path_dataset = PATH_DATASET_TWOD
        else:
            path_dataset = PATH_DATASET_THREED

        try:
            self.dataframe = pd.read_pickle(path_dataset)
        except FileNotFoundError:
            data = pd.read_pickle(PATH_DIAGNOSIS_PICKLE)
            inbalence = compute_inbalance(data)
            print("This is the imbalence")
            print(inbalence)
            print("Building the dataset")
            create_dataset(PATH_MRI, PATH_DIAGNOSIS_PICKLE, PATH_TOSAVE_DATA, TwoD)
            self.dataframe = pd.read_pickle(path_dataset)

        self.trainset_size = train_size
        self.validation_size = validation_size
        if randomize_whole_dataset:
            self.dataframe.sample(frac=1)
        rows_train = self.dataframe.index[:train_size]
        self.train = self.dataframe.ix[rows_train]
        if not randomize_whole_dataset:
            self.train.sample(frac=1)

        rows_validation = self.dataframe.index[train_size: train_size + validation_size]
        validations = self.dataframe.ix[rows_validation]
        self.validation_inputs = np.vstack(validations["mri_pickle_data"].values)
        self.validation_labels = np.vstack(validations["label"].values)

        rows_test = self.dataframe.index[train_size + validation_size:]
        tests = self.dataframe.ix[rows_test]
        self.tests_inputs = np.vstack(tests["mri_pickle_data"].values)
        self.tests_labels = np.vstack(tests["label"].values)

        self.batch_index = 0


    def train_next_batch(self, batch_size):
        if self.batch_index + batch_size < self.trainset_size:
            rows = self.train.index[self.batch_index: self.batch_index + batch_size]
            self.batch_index = self.batch_index + batch_size
            frame = self.train.ix[rows]
        else:
            rows_1 = self.train.index[self.batch_index:]
            rows_2 = self.train.index[:(self.batch_index + batch_size)%self.trainset_size]
            self.batch_index = (self.batch_index + batch_size) % self.trainset_size
            frame_1 = self.train.ix[rows_1]
            frame_2 = self.train.ix[rows_2]
            frame = frame_1.append(frame_2)
        x = np.vstack(frame["mri_pickle_data"].values)
        y = np.vstack(frame["label"].values)
        return x, y






