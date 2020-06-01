import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from test_enrichment import ML
from phe import paillier
from enricher import enrich
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


datalevels = [0,10000,20000,35000,40000,60000]
listofweights = []
no_of_iterations = 5
for iterations in range(no_of_iterations):
    listofweights.clear()
    for i in range(4):
        print('calling', i)
        weights_new = ML(x_train[datalevels[i]:datalevels[i+1],:],y_train[datalevels[i]:datalevels[i+1],:],weights_old) 
        print('encrypted weights received')
        listofweights.append(weights_new)
    count=0
    j=k=0
    print('averaging weights....')
    for j in range(784):
        for k in range(32):
            temp0 = listofweights[0][count]
            temp1 = listofweights[1][count]
            temp2 = listofweights[2][count]
            temp3 = listofweights[3][count]
            temp4 = temp0+ temp1+temp2+temp3
            count = count + 1
            weights_old[0][j][k] = temp4/5
    j=0
    for j in range(32):
            temp0 = listofweights[0][count]
            temp1 = listofweights[1][count]
            temp2 = listofweights[2][count]
            temp3 = listofweights[3][count]
            temp4 = temp0+ temp1+temp2+temp3
            count = count + 1
            weights_old[1][j] = temp4/5
    j=k=0
    for j in range(32):
        for k in range(10):
            temp0 = listofweights[0][count]
            temp1 = listofweights[1][count]
            temp2 = listofweights[2][count]
            temp3 = listofweights[3][count]
            temp4 = temp0+ temp1+temp2+temp3
            count = count + 1
            weights_old[2][j][k] = temp4/5
    j=0
    for j in range(10):
            temp0 = listofweights[0][count]
            temp1 = listofweights[1][count]
            temp2 = listofweights[2][count]
            temp3 = listofweights[3][count]
            temp4 = temp0+ temp1+temp2+temp3
            count = count + 1
            weights_old[3][j] = temp4/5
    print('testing average model')
    model2 = Sequential()
    model2.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model2.add(Dense(units=num_classes, activation='softmax'))
    model2.set_weights(weights_old)
    model2.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    loss2, accuracy2  = model2.evaluate(x_test, y_test, verbose=False)
    print(iterations)
    print(f'Test accuracy: {accuracy2:.3}')

pub_key,priv_key = paillier.generate_paillier_keypair() 
enc_values = []
for j in range(784):
    for k in range(32):
        __temp = pub_key.encrypt(float(weights_old[0][j][k]))
        enc_values.append((str(__temp.ciphertext()),__temp.exponent))
j=0
for j in range(32):
        __temp = pub_key.encrypt(float(weights_old[1][j]))
        enc_values.append((str(__temp.ciphertext()),__temp.exponent))
j=k=0
for j in range(32):
    for k in range(10):
        __temp = pub_key.encrypt(float(weights_old[2][j][k]))
        enc_values.append((str(__temp.ciphertext()),__temp.exponent))
j=0
for j in range(10):
        __temp = pub_key.encrypt(float(weights_old[3][j]))
        enc_values.append((str(__temp.ciphertext()),__temp.exponent))
#<----------------------------------------------------------------------Enrichment Code Begins ---------------------------------------------------------------------------------->
enrichment_weights = enrich(x_train[datalevels[4]:datalevels[5],:],y_train[datalevels[4]:datalevels[5],:],weights_old,pub_key,priv_key)
count = 0 
for j in range(784):
    for k in range(32):
        temp0 = paillier.EncryptedNumber(pub_key,int(enc_values[count][0]),int(enc_values[count][1]))
        temp1 = paillier.EncryptedNumber(pub_key,int(enrichment_weights[count][0]),int(enrichment_weights[count][1]))
        temp2 = temp0.__add__(temp0).__add__(temp0).__add__(temp0).__add__(temp1)
        count = count + 1
        weights_old[0][j][k] = priv_key.decrypt(temp2)
        weights_old[0][j][k] = weights_old[0][j][k]/5
j=0
for j in range(32):
        temp0 = paillier.EncryptedNumber(pub_key,int(enc_values[count][0]),int(enc_values[count][1]))
        temp1 = paillier.EncryptedNumber(pub_key,int(enrichment_weights[count][0]),int(enrichment_weights[count][1]))
        temp2 = temp0.__add__(temp0).__add__(temp0).__add__(temp0).__add__(temp1)
        count = count + 1
        weights_old[1][j] = priv_key.decrypt(temp2)
        weights_old[1][j] = weights_old[1][j]/5
j=k=0
for j in range(32):
    for k in range(10):
        temp0 = paillier.EncryptedNumber(pub_key,int(enc_values[count][0]),int(enc_values[count][1]))
        temp1 = paillier.EncryptedNumber(pub_key,int(enrichment_weights[count][0]),int(enrichment_weights[count][1]))
        temp2 = temp0.__add__(temp0).__add__(temp0).__add__(temp0).__add__(temp1)
        count = count + 1
        weights_old[2][j][k] = priv_key.decrypt(temp2)
        weights_old[2][j][k] = weights_old[2][j][k]/5
j=0
for j in range(10):
        temp0 = paillier.EncryptedNumber(pub_key,int(enc_values[count][0]),int(enc_values[count][1]))
        temp1 = paillier.EncryptedNumber(pub_key,int(enrichment_weights[count][0]),int(enrichment_weights[count][1]))
        temp2 = temp0.__add__(temp0).__add__(temp0).__add__(temp0).__add__(temp1)
        count = count + 1
        weights_old[3][j] = priv_key.decrypt(temp2)
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



j=k=0
