#this file has code to display the number in grey format
import numpy as np
from matplotlib import pyplot as plt
def visualize(X):
    image = np.array(X, dtype='float')
    pixels = image.reshape((20, 20))
    plt.imshow(pixels, cmap='gray')
    plt.show()