import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from neuralnetwork import ML

#Training and Testing between multiple parties in decrypted domain. Kind of Simulation for the federated learning.
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
history1 = model1.fit(x_train[0:10000], y_train[0:10000], batch_size=128, epochs=5, verbose=False, validation_split=.1)
loss1, accuracy1  = model1.evaluate(x_test, y_test, verbose=False)
weights1 = model1.get_weights()
print(f'Test loss: {loss1:.3}')
print(f'Test accuracy: {accuracy1:.3}')

#<------------------------------------Calling enhancer  1-------------------------------->
model2 = Sequential()
model2.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model2.add(Dense(units=num_classes, activation='softmax'))
[weights2,history2] = ML(x_train[10000:20000,:],y_train[10000:20000,:],weights1) 
model2.set_weights(weights2)

model2.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
loss2, accuracy2  = model2.evaluate(x_test, y_test, verbose=False)
print(f'Test loss: {loss2:.3}')
print(f'Test accuracy: {accuracy2:.3}')

#<------------------------------------Calling enhancer  2-------------------------------->
model3 = Sequential()
model3.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model3.add(Dense(units=num_classes, activation='softmax'))
[weights3,history3] = ML(x_train[20000:30000,:],y_train[20000:30000,:],weights2) 
model3.set_weights(weights3)

model3.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
loss3, accuracy3  = model3.evaluate(x_test, y_test, verbose=False)
print(f'Test loss: {loss3:.3}')
print(f'Test accuracy: {accuracy3:.3}')

#<------------------------------------Calling enhancer  3-------------------------------->
model4 = Sequential()
model4.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model4.add(Dense(units=num_classes, activation='softmax'))
[weights4,history4] = ML(x_train[30000:40000,:],y_train[30000:40000,:],weights3) 
model4.set_weights(weights4)

model4.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
loss4, accuracy4  = model4.evaluate(x_test, y_test, verbose=False)
print(f'Test loss: {loss4:.3}')
print(f'Test accuracy: {accuracy4:.3}')

#<------------------------------------Calling enhancer  4-------------------------------->
model5 = Sequential()
model5.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model5.add(Dense(units=num_classes, activation='softmax'))
[weights5,history5] = ML(x_train[40000:45000,:],y_train[40000:45000,:],weights4) 
model5.set_weights(weights5)

model5.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
loss5, accuracy5  = model5.evaluate(x_test, y_test, verbose=False)
print(f'Test loss: {loss5:.3}')
print(f'Test accuracy: {accuracy5:.3}')

#<------------------------------------Calling enhancer  5-------------------------------->
model6 = Sequential()
model6.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model6.add(Dense(units=num_classes, activation='softmax'))
[weights6,history6] = ML(x_train[45000:,:],y_train[45000:,:],weights5) 
model6.set_weights(weights6)

model6.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
loss6, accuracy6  = model6.evaluate(x_test, y_test, verbose=False)
print(f'Test loss: {loss6:.3}')
print(f'Test accuracy: {accuracy6:.3}')