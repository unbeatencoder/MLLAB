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

    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.set_weights(given_weights)
    # model.summary()

    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=False, validation_split=.1)
    weights = model.get_weights()
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['training', 'validation'], loc='best')
    # plt.show()
    # keras.backend.clear_session()
    # print(f'Test loss: {loss:.3}')
    # print(f'Test accuracy: {accuracy:.3}')
    return [weights,history]
