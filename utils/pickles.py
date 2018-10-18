from six.moves import cPickle as pickle
import numpy as np


def _pickle_(dataset, path_to_pickle):
    print("PICKLING")
    try:
        with open(path_to_pickle, 'wb') as f:
            pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            print("Successfully pickled at " + path_to_pickle)
    except Exception as e:
        print('Unable to save data to', path_to_pickle, ':', e)

    return


def _read_pickle_(path):
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print('Unable to process data from', path, ':', e)
        raise


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


def convert_name_to_number(string):
    number = string[3:7]
    #print(number)
    try:
        number = int(number)
    except ValueError:
        try:
            number = int(number[:3])
        except ValueError:
            number = int(number[:2])
    return number


def reformat_2D_to_1D(dataset, size_x, size_y):
  return dataset.reshape((-1, size_x * size_y)).astype(np.float32)

