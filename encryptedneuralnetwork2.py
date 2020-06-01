import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from encryptedneuralnetwork import ML
from phe import paillier
#<------------------------------------Self Training-------------------------------->
(x_train, y_train), (x_test, y_test) = mnist.load_data()
pub_key,priv_key = paillier.generate_paillier_keypair()

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
        print('calling', i)
        weights_new = ML(x_train[datalevels[i]:datalevels[i+1],:],y_train[datalevels[i]:datalevels[i+1],:],weights_old,pub_key,priv_key) 
        print('encrypted weights received')
        listofweights.append(weights_new)
    count=0
    j=k=0
    print('averaging weights....')
    for j in range(784):
        for k in range(32):
            temp0 = paillier.EncryptedNumber(pub_key,int(listofweights[0][count][0]),int(listofweights[0][count][1]))
            temp1 = paillier.EncryptedNumber(pub_key,int(listofweights[1][count][0]),int(listofweights[1][count][1]))
            temp2 = paillier.EncryptedNumber(pub_key,int(listofweights[2][count][0]),int(listofweights[2][count][1]))
            temp3 = paillier.EncryptedNumber(pub_key,int(listofweights[3][count][0]),int(listofweights[3][count][1]))
            temp4 = paillier.EncryptedNumber(pub_key,int(listofweights[4][count][0]),int(listofweights[4][count][1]))
            temp5 = temp0.__add__(temp1).__add__(temp2).__add__(temp3).__add__(temp4)

            count = count + 1
            weights_old[0][j][k] = priv_key.decrypt(temp5)
            weights_old[0][j][k] = weights_old[0][j][k]/5
    j=0
    for j in range(32):
            temp0 = paillier.EncryptedNumber(pub_key,int(listofweights[0][count][0]),int(listofweights[0][count][1]))
            temp1 = paillier.EncryptedNumber(pub_key,int(listofweights[1][count][0]),int(listofweights[1][count][1]))
            temp2 = paillier.EncryptedNumber(pub_key,int(listofweights[2][count][0]),int(listofweights[2][count][1]))
            temp3 = paillier.EncryptedNumber(pub_key,int(listofweights[3][count][0]),int(listofweights[3][count][1]))
            temp4 = paillier.EncryptedNumber(pub_key,int(listofweights[4][count][0]),int(listofweights[4][count][1]))
            temp5 = temp0.__add__(temp1).__add__(temp2).__add__(temp3).__add__(temp4)
            count = count + 1
            weights_old[1][j] = priv_key.decrypt(temp5)
            weights_old[1][j] = weights_old[1][j]/5
    j=k=0
    for j in range(32):
        for k in range(10):
            temp0 = paillier.EncryptedNumber(pub_key,int(listofweights[0][count][0]),int(listofweights[0][count][1]))
            temp1 = paillier.EncryptedNumber(pub_key,int(listofweights[1][count][0]),int(listofweights[1][count][1]))
            temp2 = paillier.EncryptedNumber(pub_key,int(listofweights[2][count][0]),int(listofweights[2][count][1]))
            temp3 = paillier.EncryptedNumber(pub_key,int(listofweights[3][count][0]),int(listofweights[3][count][1]))
            temp4 = paillier.EncryptedNumber(pub_key,int(listofweights[4][count][0]),int(listofweights[4][count][1]))
            temp5 = temp0.__add__(temp1).__add__(temp2).__add__(temp3).__add__(temp4)
            count = count + 1
            weights_old[2][j][k] = priv_key.decrypt(temp5)
            weights_old[2][j][k] = weights_old[2][j][k]/5
    j=0
    for j in range(10):
            temp0 = paillier.EncryptedNumber(pub_key,int(listofweights[0][count][0]),int(listofweights[0][count][1]))
            temp1 = paillier.EncryptedNumber(pub_key,int(listofweights[1][count][0]),int(listofweights[1][count][1]))
            temp2 = paillier.EncryptedNumber(pub_key,int(listofweights[2][count][0]),int(listofweights[2][count][1]))
            temp3 = paillier.EncryptedNumber(pub_key,int(listofweights[3][count][0]),int(listofweights[3][count][1]))
            temp4 = paillier.EncryptedNumber(pub_key,int(listofweights[4][count][0]),int(listofweights[4][count][1]))
            temp5 = temp0.__add__(temp1).__add__(temp2).__add__(temp3).__add__(temp4)
            count = count + 1
            weights_old[3][j] = priv_key.decrypt(temp5)
            weights_old[3][j] = weights_old[3][j]/5

    print('testing average model')
    model2 = Sequential()
    model2.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model2.add(Dense(units=num_classes, activation='softmax'))
    model2.set_weights(weights_old)
    model2.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    loss2, accuracy2  = model2.evaluate(x_test, y_test, verbose=False)
    print(iterations)
    print(f'Test accuracy: {accuracy2:.3}')