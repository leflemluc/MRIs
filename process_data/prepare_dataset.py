import numpy as np
from os import listdir
import pandas as pd
from utils.pickles import _pickle_, convert_name_to_number
import nibabel as nib

ONE_HOT_DIMENSION = 5

DIMENSION = 256

PATH_TOSAVE_DATA = "/Users/lucleflem/Desktop/Ali/data/full_dataset.p"
PATH_MRI = "/Users/lucleflem/Desktop/Ali/data/NACC_small/"
PATH_DIAGNOSIS = "/Users/lucleflem/Desktop/Ali/data/data_nacc_diagnosis.xlsx"
PATH_DIAGNOSIS_PICKLE = "/Users/lucleflem/Desktop/Ali/data/data_nacc_diagnosis.p"


def pandize_diagnosis_file(path_diagnosis_pickle, path_diagnosis):
    print("Reading")
    data = pd.read_excel(path_diagnosis)
    print("PROCESSING")
    data["NACCMRFI"] = data["NACCMRFI"].apply(lambda x: x[3:-4])
    print("PICKLING")
    data.to_pickle(path_diagnosis_pickle)

def compute_inbalance(data):
    inbalance = pd.DataFrame(columns=["0", "1", "2", "3", "4"])
    N = len(data)
    for i in inbalance.columns:
        inbalance[i] = [len(data[data["diagnosis_0normal_1normaltomci_2mci_3ad_4otherdementia"]==int(i)]) / N]
    return inbalance


def create_dataset(path_mri, path_diagnosis, path_dataset, TwoD):
    data_diagnosis = pd.read_pickle(path_diagnosis)
    dataset = pd.DataFrame(columns=["mri_pickle_data", "label"])
    index_dataset = 0
    for f in listdir(path_mri):
        index_dataset+=1
        print(f)
        mri_index = f[3:7]
        try:
            index = int(mri_index)
        except ValueError:
            mri_index = f[3:6]
            try:
                index = int(mri_index)
            except ValueError:
                mri_index = f[3:5]
        line = data_diagnosis[data_diagnosis["NACCMRFI"] == mri_index]
        if not line.empty:
            img = nib.load(path_mri + f)
            assert img.shape == (DIMENSION, DIMENSION, DIMENSION)
            data_mri = img.get_fdata()
            if not TwoD:
                data_mri = np.reshape(data_mri, DIMENSION * DIMENSION * DIMENSION)
            else:
                cut_x = data_mri.shape[0] // 2
                cut_y = data_mri.shape[1] // 2
                cut_z = data_mri.shape[2] // 2
                slice_0 = np.reshape(data_mri[cut_x, :, :], DIMENSION * DIMENSION)
                slice_1 = np.reshape(data_mri[:, cut_y, :], DIMENSION * DIMENSION)
                slice_2 = np.reshape(data_mri[:, :, cut_z], DIMENSION * DIMENSION)
                data_mri = np.append(slice_0, np.append(slice_1, slice_2))
            label = np.zeros(shape=5)
            label[line["diagnosis_0normal_1normaltomci_2mci_3ad_4otherdementia"].values[0]] = 1
            df = pd.DataFrame({"mri_pickle_data": [data_mri], 'label': [label]})
            dataset = dataset.append(df, ignore_index=True)
    print("The number of indices is " + str(index_dataset))
    dataset.to_pickle(path_dataset)
    return dataset




if __name__=='__main__':

    data = pd.read_pickle(PATH_DIAGNOSIS_PICKLE)
    inbalence = compute_inbalance(data)
    print("This is the imbalence")
    print(inbalence)
    print("Building the dataset")
    dataset = create_dataset(PATH_MRI, PATH_DIAGNOSIS_PICKLE, PATH_TOSAVE_DATA)




