import keras
import matplotlib.pyplot as plt
import json
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
# Setup train and test splits
from phe import paillier

#Simulates the role of enricher.
# In real life, we won't need a priv_key if we are doing enrichment in Encrypted Domain.
#Hopefully, we will have a library soon... :)
def enrich(x_train,y_train,given_weights,pub_key,priv_key):
    print('Enricher called....')
    #Initialise Keras Model.
    image_size = 784 # 28*28
    num_classes = 10 # ten unique digits
    model = Sequential()
    model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='softmax'))

    #Set weights that are result of the collaborative training.
    # We want this to be encrypted domain if it is possible to do so.
    model.set_weights(given_weights)

    #Training on the local data of the enricher.    
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=False, validation_split=.1)
    weights = model.get_weights()
    j=k=0
    print('local training done, encrypting gradients')
    enc_values = []

    #Send encrypted weights to the caller.
    #In real life, these would be uploaded to blockchain.
    # Collaborative party shouldn't be able to decrypt these values.
    #If the compensation will be revenue based, key handover will occur when reward is granted 
    #If the compensation will be profit share based, key handover will not occur and the gradients will remain in 
    # encrypted domain.
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
    return enc_values
