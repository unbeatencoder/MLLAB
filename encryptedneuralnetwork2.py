import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from encryptedneuralnetwork import ML
#<------------------------------------Self Training-------------------------------->
(x_train, y_train), (x_test, y_test) = mnist.load_data()


image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


image_size = 784
num_classes = 10

model1 = Sequential()
model1.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model1.add(Dense(units=num_classes, activation='softmax'))
model1.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(x_train[0:100], y_train[0:100], batch_size=128, epochs=1, verbose=False, validation_split=.1)
loss1, accuracy1  = model1.evaluate(x_test, y_test, verbose=False)
weights_old = model1.get_weights()
print(f'Test loss: {loss1:.3}')
print(f'Test accuracy: {accuracy1:.3}')


datalevels = [0,10000,20000,35000,50000,60000]
listofweights = []
for iterations in range(10):
    listofweights.clear()
    for i in range(5):
        weights_new = ML(x_train[datalevels[i]:datalevels[i+1],:],y_train[datalevels[i]:datalevels[i+1],:],weights_old) 
        listofweights.append(weights_new)

    j=k=0
    for j in range(784):
        for k in range(32):
            weights_old[0][j][k] = (listofweights[0][0][j][k]+listofweights[1][0][j][k]+listofweights[2][0][j][k]+listofweights[3][0][j][k]+listofweights[4][0][j][k])/5
    j=0
    for j in range(32):
            weights_old[1][j] = (listofweights[0][1][j]+listofweights[1][1][j]+listofweights[2][1][j]+listofweights[3][1][j]+listofweights[4][1][j])/5
    j=0
    for j in range(32):
        for k in range(10):
            weights_old[2][j][k] = (listofweights[0][2][j][k]+listofweights[1][2][j][k]+listofweights[2][2][j][k]+listofweights[3][2][j][k]+listofweights[4][2][j][k])/5
    j=0
    for j in range(10):
            weights_old[3][j] = (listofweights[0][3][j]+listofweights[1][3][j]+listofweights[2][3][j]+listofweights[3][3][j]+listofweights[4][3][j])/5
    model2 = Sequential()
    model2.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model2.add(Dense(units=num_classes, activation='softmax'))
    model2.set_weights(weights_old)
    model2.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    loss2, accuracy2  = model2.evaluate(x_test, y_test, verbose=False)
    print(iterations)
    print(f'Test accuracy: {accuracy2:.3}')
