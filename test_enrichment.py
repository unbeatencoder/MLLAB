import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
# Setup train and test splits
def ML(x_train,y_train,given_weights):

    image_size = 784 # 28*28
    num_classes = 10 # ten unique digits
    model = Sequential()
    model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.set_weights(given_weights)

    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=False, validation_split=.1)
    weights = model.get_weights()
    return [weights,history]
