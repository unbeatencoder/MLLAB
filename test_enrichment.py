import keras
import matplotlib.pyplot as plt
import json
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
# Setup train and test splits
from phe import paillier

#This function mimics one iteration of an enricher for training MNIST dataset. 
# It receives global model, then it trains the global model with it's local data
# and then sends the gradients and bias in the encrypted format to the Blockchain.
#@x_train : This is it's local data which for stimulation is passed by the caller,
# by distributing the MNIST data set into equal partitions. But, in real life it will 
# be the owner themselves who have their data. 
#@y_train : This is it's local labels corresponding to each set of features in the x_train 
# which for stimulation is passed by the caller, by distributing the MNIST data set into equal 
# partitions. But, in real life it will be the owner themselves who have their data. 
#@given_weights : This is the global model which was result of either previous iterations
#calculated by averaging gradients of all the participants or if it's first iteration, then 
# the value of the initial model as decided by collaborative parties.
#@pub_key : Key to encrypt the gradients and bias after local training is completed.
#@priv_key : Key to decrypt the gradients downloaded from the blockchain which is the global 
#view of the model. However, in real life, this key won't be available to the party. This secret key
#will be distributed between the parties and enricher using t-of-n shamir secret sharing key.
#Note : In real life, we aim that encricher trains the model in encrypted domain. But, due to lack of 
#encrypted neural network training libraries, we are doing it in the decrypted domain. Hence, here the training 
# and improvement will happen in decrypted domain.
def ML(x_train,y_train,given_weights):
    print('ML called....')

    image_size = 784 # 28*28
    num_classes = 10 # ten unique digits

    #Create a model as necessary using keras library.
    model = Sequential()
    model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    #Set weights of this model same as that of global model.
    model.set_weights(given_weights)

    #Train the global view of model with your local data.
    model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=False, validation_split=.1)

    #get weights and bias of the locally trained model.
    weights = model.get_weights()
    j=k=0
    print('local training done, encrypting gradients')

    #Encrypt values and send to the caller.
    #This looks different as we are not encrypting the values, but this is to save compuation for the sake of simulation.
    #In real life before enrichment, this values will be uploaded to blockchain using a smart contract.
    enc_values = []
    for j in range(784):
        for k in range(32):
            enc_values.append(float(weights[0][j][k]))
    j=0
    for j in range(32):
            enc_values.append(float(weights[1][j]))

    j=k=0
    for j in range(32):
        for k in range(10):
            enc_values.append(float(weights[2][j][k]))
    j=0
    for j in range(10):
            enc_values.append(float(weights[3][j]))
            
    print('sending encrypted gradients')
    #Append and Return.
    return enc_values
