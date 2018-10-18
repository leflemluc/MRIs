from matplotlib import pyplot as plt
import numpy as np

def gen_image(arr):
    X = arr.reshape([28, 28])
    plt.gray()
    plt.imshow(X)
    """
    two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
    plt.imshow(two_d, interpolation='nearest')
    """
    return plt
