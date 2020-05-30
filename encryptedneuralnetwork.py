import keras
import matplotlib.pyplot as plt
import json
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
# Setup train and test splits
from phe import paillier

def ML(x_train,y_train,given_weights,pub_key,priv_key):
    print('ML called....')
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Flatten the images
    # image_vector_size = 28*28
    # x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    # x_test = x_test.reshape(x_test.shape[0], image_vector_size)

    # num_classes = 10
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    # print(x_train.shape)
    # print(x_test.shape)
    # x_train = x_train[0:20000,:]
    # y_train = y_train[0:20000,:]
    
    # print("Training data shape: ", x_train.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels
    # print("Test data shape", y_train.shape) # (10000, 28, 28) -- 10000 images, each 28x28

    image_size = 784 # 28*28
    num_classes = 10 # ten unique digits
    model = Sequential()
    model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.set_weights(given_weights)

    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=False, validation_split=.1)
    weights = model.get_weights()
    j=k=0
    print('local training done, encrypting gradients')
    enc_values = []
    for j in range(784):
        for k in range(32):
            __temp = pub_key.encrypt(float(weights[0][j][k]))
            enc_values.append((str(__temp.ciphertext()),__temp.exponent))
    j=0
    for j in range(32):
            __temp = pub_key.encrypt(float(weights[1][j]))
            enc_values.append((str(__temp.ciphertext()),__temp.exponent))

    j=k=0
    for j in range(32):
        for k in range(10):
            __temp = pub_key.encrypt(float(weights[2][j][k]))
            enc_values.append((str(__temp.ciphertext()),__temp.exponent))
    j=0
    for j in range(10):
            __temp = pub_key.encrypt(float(weights[3][j]))
            enc_values.append( (str(__temp.ciphertext()),__temp.exponent))
    print('sending encrypted gradients')
#     print(f'Test loss: {loss:.3}')
#     print(f'Test accuracy: {accuracy:.3}')


    return enc_values
