import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # Documentation: https://keras.io/models/sequential/
from encryptedneuralnetwork import ML
from phe import paillier



#Consider this program as that which will run on the blockchain. Following are the major steps :
#1. Initialise model with an consensus.
#2. For each iteration, send the global model to participants, receive their gradients and aggregate them.
#3. Have a test data, predict using the updated model and expect accuracy increases every time or at least should not decrease.
#<------------------------------------Self Training-------------------------------->

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

#This datalevels describes the distribution of data between 5 collaborative parties.
# i th party owns data from MNIST dataset starting with index i-1 to i where i starts from 1 and goes to 5.
#Again, in real life, we won't need this. each party will have their own data. This stimulates that each party 
# has different dataset and we will try to look at the result of the aggregate model at the end of stimulation.
datalevels = [0,10000,20000,35000,50000,60000]
listofweights = []

#We will do 10 iterations of training.
#For each iteration, this program will call ML() function which represents one collaborative party.
# Each party will receive global model which is sent using @weight_old.
# @x_train, @y_train : Their training data which as described they would already have in real life.
# @pub_key : To encrypt the parameters : gradients and biases after the local training iteration.
# @priv_key : Again, this is sent because it is difficult to stimulate a distributed decryption of the global model.
for iterations in range(10):
    listofweights.clear()
    for i in range(5):
        print('calling', i)
        #sending global model and receiving the data from collaborative party i after one iteration of local training on the data with i th party.
        weights_new = ML(x_train[datalevels[i]:datalevels[i+1],:],y_train[datalevels[i]:datalevels[i+1],:],weights_old,pub_key,priv_key) 
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
            #Retrieving encrypted number from the basis and exponents received from each party.
            temp0 = paillier.EncryptedNumber(pub_key,int(listofweights[0][count][0]),int(listofweights[0][count][1]))
            temp1 = paillier.EncryptedNumber(pub_key,int(listofweights[1][count][0]),int(listofweights[1][count][1]))
            temp2 = paillier.EncryptedNumber(pub_key,int(listofweights[2][count][0]),int(listofweights[2][count][1]))
            temp3 = paillier.EncryptedNumber(pub_key,int(listofweights[3][count][0]),int(listofweights[3][count][1]))
            temp4 = paillier.EncryptedNumber(pub_key,int(listofweights[4][count][0]),int(listofweights[4][count][1]))
            #Add them. 
            temp5 = temp0.__add__(temp1).__add__(temp2).__add__(temp3).__add__(temp4)
            count = count + 1
            #Decrypt the values.
            #Ideally, in real life this computation will not occur on the blockchain.
            #Each party will share with each other their decryption shares to get the decrypted values of the params.
            #Even in the simulation, this computation should be first part of the function which simulates the collaborative 
            # party because we are giving them private key. 
            # But, the idea here is to optimise the simulation cause it doesn't make sense to decrypt same values 5 times 
            # for each participants just for the simulation. 
            weights_old[0][j][k] = priv_key.decrypt(temp5)
            weights_old[0][j][k] = weights_old[0][j][k]/5
    
    #For bias of the first layer of 32 neurons.
    j=0
    for j in range(32):
            #Retrieving encrypted number from the basis and exponents received from each party.
            temp0 = paillier.EncryptedNumber(pub_key,int(listofweights[0][count][0]),int(listofweights[0][count][1]))
            temp1 = paillier.EncryptedNumber(pub_key,int(listofweights[1][count][0]),int(listofweights[1][count][1]))
            temp2 = paillier.EncryptedNumber(pub_key,int(listofweights[2][count][0]),int(listofweights[2][count][1]))
            temp3 = paillier.EncryptedNumber(pub_key,int(listofweights[3][count][0]),int(listofweights[3][count][1]))
            temp4 = paillier.EncryptedNumber(pub_key,int(listofweights[4][count][0]),int(listofweights[4][count][1]))
            #Add them.
            temp5 = temp0.__add__(temp1).__add__(temp2).__add__(temp3).__add__(temp4)
            count = count + 1
            #Decrypt the values.
            #Ideally, in real life this computation will not occur on the blockchain.
            #Each party will share with each other their decryption shares to get the decrypted values of the params.
            #Even in the simulation, this computation should be first part of the function which simulates the collaborative 
            # party because we are giving them private key. 
            # But, the idea here is to optimise the simulation cause it doesn't make sense to decrypt same values 5 times 
            # for each participants just for the simulation. 
            weights_old[1][j] = priv_key.decrypt(temp5)
            weights_old[1][j] = weights_old[1][j]/5
    
    #For gradients of output layer which has 10 neurons.
    j=k=0
    for j in range(32):
        for k in range(10):
            #Retrieving encrypted number from the basis and exponents received from each party.
            temp0 = paillier.EncryptedNumber(pub_key,int(listofweights[0][count][0]),int(listofweights[0][count][1]))
            temp1 = paillier.EncryptedNumber(pub_key,int(listofweights[1][count][0]),int(listofweights[1][count][1]))
            temp2 = paillier.EncryptedNumber(pub_key,int(listofweights[2][count][0]),int(listofweights[2][count][1]))
            temp3 = paillier.EncryptedNumber(pub_key,int(listofweights[3][count][0]),int(listofweights[3][count][1]))
            temp4 = paillier.EncryptedNumber(pub_key,int(listofweights[4][count][0]),int(listofweights[4][count][1]))
            #Add them.
            temp5 = temp0.__add__(temp1).__add__(temp2).__add__(temp3).__add__(temp4)
            count = count + 1
            #Decrypt the values.
            #Ideally, in real life this computation will not occur on the blockchain.
            #Each party will share with each other their decryption shares to get the decrypted values of the params.
            #Even in the simulation, this computation should be first part of the function which simulates the collaborative 
            # party because we are giving them private key. 
            # But, the idea here is to optimise the simulation cause it doesn't make sense to decrypt same values 5 times 
            # for each participants just for the simulation.
            weights_old[2][j][k] = priv_key.decrypt(temp5)
            weights_old[2][j][k] = weights_old[2][j][k]/5
    
    #For bias of the output layer neurons.
    j=0
    for j in range(10):
            #Retrieving encrypted number from the basis and exponents received from each party.
            temp0 = paillier.EncryptedNumber(pub_key,int(listofweights[0][count][0]),int(listofweights[0][count][1]))
            temp1 = paillier.EncryptedNumber(pub_key,int(listofweights[1][count][0]),int(listofweights[1][count][1]))
            temp2 = paillier.EncryptedNumber(pub_key,int(listofweights[2][count][0]),int(listofweights[2][count][1]))
            temp3 = paillier.EncryptedNumber(pub_key,int(listofweights[3][count][0]),int(listofweights[3][count][1]))
            temp4 = paillier.EncryptedNumber(pub_key,int(listofweights[4][count][0]),int(listofweights[4][count][1]))
            #Add them.
            temp5 = temp0.__add__(temp1).__add__(temp2).__add__(temp3).__add__(temp4)
            count = count + 1
            #Decrypt the values.
            #Ideally, in real life this computation will not occur on the blockchain.
            #Each party will share with each other their decryption shares to get the decrypted values of the params.
            #Even in the simulation, this computation should be first part of the function which simulates the collaborative 
            # party because we are giving them private key. 
            # But, the idea here is to optimise the simulation cause it doesn't make sense to decrypt same values 5 times 
            # for each participants just for the simulation.
            weights_old[3][j] = priv_key.decrypt(temp5)
            weights_old[3][j] = weights_old[3][j]/5

    #Compute accuracy on the test data using the aggregated model.
    # This should ideally happen entirely in the encrypted domain. 
    # Since, there is lack of approaches for encrypted neural network training and evaluation, 
    # we are doing this in decrypted domain. In the mean time other researchers look for encrypted nerual network training,
    # All the parties are anyways going to decrypt gradients before training of next iteration[unless in private mode], 
    # they can evaluate themselves with their local test data and keep checking if training is going in the right direction. 
    print('testing average model')
    model2 = Sequential()
    model2.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
    model2.add(Dense(units=num_classes, activation='softmax'))
    #Set weights as aggregated from the collaborative parties.
    model2.set_weights(weights_old)
    #Compile model and evaluate.
    model2.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
    loss2, accuracy2  = model2.evaluate(x_test, y_test, verbose=False)
    print(iterations)
    #Print Accuracy. This should go up with training.
    print(f'Test accuracy: {accuracy2:.3}')