import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from test_enrichment import ML
from phe import paillier
from enricher import enrich


#Consider this program as that which will run on the blockchain. Following are the major steps :
#1. Send the encrypted gradients to the enricher.
#2. enricher trains on it's own data and uploads to blockchain.
#3. Rewards enricher based on the accuracy vs compensation plan defined by consensus between parties.
#Note : We assume that transfer of key has already happened and that all the gradients are stored on blockchain in encrypted format.



#<------------------------------------Self Training-------------------------------->

#Setup to reach the stage where the Blockchain already has the encrypted gradients through training between collaborative group.
# We do this by using dividing MNIST dataset into 5 parts. First we only train with 4 parts of this data and we intentionaly leave
# out the 5 th portion for the enricher. 

#1. Initialise the Model Parameters. Ideally, this will be done distributedly and offline by 
# reaching a consensus between the collaborative parties.

#Load Train and Test Data. 
#In real life, the local training data will be available to each of participants from whichever source they collect.
#In real life, the testing data will be collected in encrypted format from each of the participants.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Generate the public and priv key pair of homomorphic encryption. 
#In real life, this process will be done by a trusted vendor or using distributed mechanism to generate a public key
# and distribute secret key between participants based on the "t" in the t-of-n shamir secret sharing scheme.
pub_key,priv_key = paillier.generate_paillier_keypair()

#Setting up model params for the keras model.
image_vector_size = 28*28
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


image_size = 784
num_classes = 10

#Training initial model with just 100 data points.
#In real life, this will be the initial model decided by the consensus between collaborative parties.
model1 = Sequential()
model1.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model1.add(Dense(units=num_classes, activation='softmax'))
model1.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
model1.fit(x_train[0:100], y_train[0:100], batch_size=128, epochs=1, verbose=False, validation_split=.1)
loss1, accuracy1  = model1.evaluate(x_test, y_test, verbose=False)
weights_old = model1.get_weights()
print(f'Test loss: {loss1:.3}')
print(f'Test accuracy: {accuracy1:.3}')

#This datalevels describes the distribution of data between 4 collaborative party and last portion is reserved 
# for the enricher stimulation.
# i th party owns data from MNIST dataset starting with index i-1 to i where i starts from 1 and goes to 4.
# Enricher owns data from 40000-60000.
#Again, in real life, we won't need this. each party will have their own data. This stimulates that each party 
# has different dataset and we will try to look at the result of the aggregate model at the end of stimulation.
datalevels = [0,10000,20000,35000,40000,60000]
listofweights = []
no_of_iterations = 5


#We will do 10 iterations of training.
#For each iteration, this program will call ML() function which represents one collaborative party.
# Each party will receive global model which is sent using @weight_old.
# @x_train, @y_train : Their training data which as described they would already have in real life.
# @pub_key : To encrypt the parameters : gradients and biases after the local training iteration.
# @priv_key : Again, this is sent because it is difficult to stimulate a distributed decryption of the global model.
#At the end of these iterations, we will have a model in encrypted domain stored on the Blockchain. 
# Skip through this code and refer to encryptedneuralnetwork2 to understand this process in real essence.
# Here, we are not aggregating gradients or receiving gradients in encrypted domain because this does not matter in stimulation
# of the enrichment. This stage will be reached before enrichment starts.
for iterations in range(no_of_iterations):
    listofweights.clear()
    for i in range(4):
        print('calling', i)
        #sending global model and receiving the data from collaborative party i after one iteration of local training on the data with i th party.
        weights_new = ML(x_train[datalevels[i]:datalevels[i+1],:],y_train[datalevels[i]:datalevels[i+1],:],weights_old) 
        print('encrypted weights received')
        #Append the weights to a list of weight of this iteration.
        listofweights.append(weights_new)
    count=0
    j=k=0
    print('averaging weights....')

    #Aggregate these weights to compute the global view of the params : gradients and biases.

    #For gradients of the input layer i.e. 784 : input and 32 : no of neurons in first layer.
    for j in range(784):
        for k in range(32):
            #Retrieving  number from the basis and exponents received from each party.
            temp0 = listofweights[0][count]
            temp1 = listofweights[1][count]
            temp2 = listofweights[2][count]
            temp3 = listofweights[3][count]
            temp4 = temp0+ temp1+temp2+temp3
            count = count + 1
            weights_old[0][j][k] = temp4/5
    j=0
    #For bias of the first layer of 32 neurons.
    for j in range(32):
            temp0 = listofweights[0][count]
            temp1 = listofweights[1][count]
            temp2 = listofweights[2][count]
            temp3 = listofweights[3][count]
            temp4 = temp0+ temp1+temp2+temp3
            count = count + 1
            weights_old[1][j] = temp4/5
    j=k=0
    #For gradients of output layer which has 10 neurons.
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
    #For bias of the output layer neurons.
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

#Gerate a public_private_key pairs of t-of-n shamir secret sharing homomorphic encryption scheme.
#Encrypt gradients and the biases of the Model. 
# Again, this stage will be reached before enrichment starts as part of collaborative training between parties.
# Refer to encryptedneuralnetwork2.py to see how it will work in reality.
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
#So, far we already have parameters of a model stored on the Blockchain in encrypted model with Key Pair -1.
# Now, enricher will aproach the collaborative group to train enrich their model and they will reach a consensus on the 
# compenstation relative the amount of accuracy the enricher improves for their model.



#Before enrichment begins, we will transfer the key with which data is encrypted to another key
# of type t-of-n where enricher also has the share of the decryption key using the mechanism defined in the transfer_key.py code. 
# here we aren't doing that for sake of simulations.


#Call the enricher to enrich our model.
#@param x_train,y_train : data for training. In real life, enricher will already have this data.
#@weights_old : These are encrypted weights we got using the collaborative training.
#@pub_key : This might be required and is already part of public knowledge.
#@priv_key : This is not required in real life as the enrichment training will occur in encrypted domain. Due to lack of libraries,
# we are doing enrichment in the decrypted domain for simulation.
enrichment_weights = enrich(x_train[datalevels[4]:datalevels[5],:],y_train[datalevels[4]:datalevels[5],:],weights_old,pub_key,priv_key)
count = 0 

#Decrypt Gradients to evaluate the model accuracy. 
#Ideally, we want this evaluation to happen in encrypted domain. Due to lack of libraries for encrypted training and testing of 
# nerual networks, we are doing enrichment in the decrypted domain for simulation. 
for j in range(784):
    for k in range(32):
        temp0 = paillier.EncryptedNumber(pub_key,int(enc_values[count][0]),int(enc_values[count][1]))
        temp1 = paillier.EncryptedNumber(pub_key,int(enrichment_weights[count][0]),int(enrichment_weights[count][1]))
        #Here, we are taking weighted average of the model ie adding temp0 4 times and new value once.
        #Again, we want all of this in encrypted domain as soon as we have a library for the encrypted neural network training
        # and testing.
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

#Test the enriched model.
print('testing average model')
model2 = Sequential()
model2.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model2.add(Dense(units=num_classes, activation='softmax'))
#Set weights that are received after enrichment.
model2.set_weights(weights_old)
model2.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
loss2, accuracy2  = model2.evaluate(x_test, y_test, verbose=False)
print(iterations)
#Print Accuracy
print(f'Test accuracy: {accuracy2:.3}')
#Grant rewards based on pre-agreed contract of accuracy improvement vs rewards.
print('You are rewarded with amount x or profit share y as part of your enrichment contribution.')



j=k=0
