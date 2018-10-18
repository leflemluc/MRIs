import pandas as pd
import numpy as np
import nibabel as nib
from process_data.prepare_dataset_V2 import PATH_FINAL_DATASET, PATH_DIAGNOSIS_PICKLE, PATH_MRI, DIMENSION
from process_data.prepare_dataset_V2 import create_dataset, compute_inbalance




class Dataset:
    def __init__(self, train_size, validation_size, test_size, randomize_whole_dataset):
        try:
            print("pickling dataset")
            self.dataframe = pd.read_pickle(PATH_FINAL_DATASET)
        except FileNotFoundError:
            data = pd.read_pickle(PATH_DIAGNOSIS_PICKLE)
            inbalence = compute_inbalance(data)
            print("This is the imbalence")
            print(inbalence)
            print("Building the dataset")
            create_dataset(PATH_MRI, PATH_DIAGNOSIS_PICKLE, PATH_FINAL_DATASET)
            self.dataframe = pd.read_pickle(PATH_FINAL_DATASET)

        self.trainset_size = train_size
        self.validation_size = validation_size
        if randomize_whole_dataset:
            self.dataframe.sample(frac=1)
        rows_train = self.dataframe.index[:train_size]
        self.train = self.dataframe.ix[rows_train]
        if not randomize_whole_dataset:
            self.train.sample(frac=1)


        print("Preparing validation set")
        rows_validation = self.dataframe.index[train_size: train_size + validation_size]
        validations = self.dataframe.ix[rows_validation]
        dataset = pd.DataFrame(columns=["mri_pickle_data", "label"])
        for i in validations.index:
            print("loading " + validations["mri_url_data"][i])
            img = nib.load(validations["mri_url_data"][i])
            assert img.shape == (DIMENSION, DIMENSION, DIMENSION)
            data_mri = img.get_fdata()
            data_mri = np.reshape(data_mri, DIMENSION * DIMENSION * DIMENSION)
            label = validations["label"][i]
            df = pd.DataFrame({"mri_pickle_data": [data_mri], 'label': [label]})
            dataset = dataset.append(df, ignore_index=True)
        self.validation_inputs = np.vstack(dataset["mri_pickle_data"].values)
        self.validation_labels = np.vstack(dataset["label"].values)

        print("Preparing test set")
        rows_test = self.dataframe.index[train_size + validation_size:train_size + validation_size + test_size]
        tests = self.dataframe.ix[rows_test]
        dataset = pd.DataFrame(columns=["mri_pickle_data", "label"])
        for i in tests.index:
            print("loading " + tests["mri_url_data"][i])
            img = nib.load(tests["mri_url_data"][i])
            assert img.shape == (DIMENSION, DIMENSION, DIMENSION)
            data_mri = img.get_fdata()
            data_mri = np.reshape(data_mri, DIMENSION * DIMENSION * DIMENSION)
            label = tests["label"][i]
            df = pd.DataFrame({"mri_pickle_data": [data_mri], 'label': [label]})
            dataset = dataset.append(df, ignore_index=True)
        self.tests_inputs = np.vstack(dataset["mri_pickle_data"].values)
        self.tests_labels = np.vstack(dataset["label"].values)

        self.batch_index = 0


    def train_next_batch(self, batch_size):
        print("Next batch")
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
        dataset_batch = pd.DataFrame(columns=["mri_pickle_data", "label"])
        for i in frame.index:
            print("loading " + frame["mri_url_data"][i])
            img = nib.load(frame["mri_url_data"][i])
            assert img.shape == (DIMENSION, DIMENSION, DIMENSION)
            data_mri = img.get_fdata()
            data_mri = np.reshape(data_mri, DIMENSION * DIMENSION * DIMENSION)
            label = frame["label"][i]
            df = pd.DataFrame({"mri_pickle_data": [data_mri], 'label': [label]})
            dataset_batch = dataset_batch.append(df, ignore_index=True)
        x = np.vstack(dataset_batch["mri_pickle_data"].values)
        y = np.vstack(dataset_batch["label"].values)
        return x, y






