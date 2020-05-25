from parameters import input_layer_size, num_labels
from loadData import getData
from CostFunction import CostFunc
from visualize import visual
from onevsAll import onevsAllAlgo
# import tensorflow as tf
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

# visual(data_x[3001])

print('Testing Regualarized Logistic Regression Cost Function')
# theta_initial=np.array([[-2],[-1],[1], [2]])
# X_temp = np.array([[1,0.1,0.6,1.1],
#           [1,0.2,0.7,1.2],
#           [1,0.3,0.8,1.3],
#           [1,0.4,0.9,1.4],
#           [1,0.5,1.0,1.5]])
# y_temp = np.array([[1],[0],[1],[0],[1]])
# reg = 3
# print(CostFunc(theta_initial,X_temp,y_temp,reg))

print('Running OnevsAll')
print(data_y)
lam = 0.1
finalTheta = onevsAllAlgo(data_x,data_y,num_labels,lam)
print(finalTheta.shape)







