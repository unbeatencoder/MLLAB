from parameters import input_layer_size, num_labels 
from loadData import getData
from visualize import visualize 
import tensorflow as tf
import numpy as np
# print(input_layer_size)
# print(num_labels)

print('Loading and Visualizing Data....')
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

data_x,data_y=getData()
# print(data_x.shape)
# print(data_y.shape)
# print(data_y[3001])
# visualize(data_x[3001])






