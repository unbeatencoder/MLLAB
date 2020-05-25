# This file contains the code to import data from the .mat file format
from scipy.io import loadmat
import pandas as pd

#loads data using loadmat function

def getData():
    data = loadmat('MNIST.mat')
    # print(type(data['X']),data['X'].shape)
    # print(type(data['y']),data['y'].shape)
    # print(type(data['X'][0][0]),data['X'][0][0].shape)
    # print(type(data['y'][0][0]),data['y'][0][0].shape)

    #using pd library to get arrays from the dictionary structure
    train_data_x = pd.DataFrame(data['X'])
    train_data_y = pd.DataFrame(data['y'])
    array_x = train_data_x.to_numpy()
    array_y = train_data_y.to_numpy()
    # print(train_data_x.shape)
    # print(train_data_y.shape)
    # print(train_data_x[0][399])
    # print(train_data_y[0][0])
    return array_x,array_y

