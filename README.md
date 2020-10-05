Project : 
This is implementation of project that I worked on during the final semester as part of my B.Tech requirement.
Following is the abstract of my BTP Report : 

Abstract—Deep Learning has been proved to be one the most
efficient Machine Learning algorithms across a wide variety of
applications in sectors like Healthcare , Education, Banking, etc.
However, Deep Learning demands a huge amount of data and
computational power during the training phase. In order to avoid
collecting data and concentrating computational power at a single
node, efforts are being made to make this process decentralized.
Moreover, a decentralized approach allows a group of people to
come together and train models, rather than limiting such a great
asset in the reach of huge corporations. But, such ambitious work
comes with its own set of challenges such as maintaining privacy
of data, Auditability between multiple parties and Fair incentives
everyone. Federated Learning is one of the most promising
solutions for such a decentralized approach. However, it has
some privacy and security drawbacks. With the rise in demand
of prediction services, a prediction as a service model is also
evolving with time. In this paper, we present a Privacy Preserving
Distributed Deep Learning framework based on Blockchain with
features of model enhancement and a decentralized prediction as
a service mechanism for a trust-less environment.


I will add link to the entire paper as soon as it is accepted in one of the conferences and available for public after conference presentation.

Note : 
First and Foremost, As soon as we have an library which can do training and testing of the encrypted neural network, we can realise the thoughts and ideas of this paper in real life. This limitation stops us from implementing the private mode of training, enrichment process that we would like to have and prediction as a service. I am actively observing development in this field, and if you know or have a library, please let me know :)

Secondly, this is just a prototype logic for what we aim to do. As soon as we are able to move to Blockchain, a lot of logic will be decoupled into different smart contracts who interact with each other based on the event triggers.



==> There are 12 python files in this repository :
1. main.py : This program tests whether addition done in encrypted domain works properly.
2. neuralnetwork.py : This program trains neural network in neural network using keras library.
3. neuralnetwork2.py : Training and Testing between multiple parties in decrypted domain using keras library. It is kind of simulation for the federated learning algorithm except here each function call refers to the one iteration of training by a single party.
4. encryptedneuralnetwork.py : This function mimics one iteration of a single party for training MNIST dataset. 
    It receives global model, then it trains the global model with it's local data
    and then sends the gradients and bias in the encrypted format to the Blockchain.
5. encryptedneuralnetwork2.py : Consider this program as that which will run on the blockchain. It performs orchestration for our idea of flow instead of event driven smart contracts that will replace this flow on the Blockchain. Following are the major steps :
    1. Initialise model with an consensus.
    2. For each iteration, send the global model to participants, receive their gradients and aggregate them.
    3. Have a test data, predict using the updated model and expect accuracy increases every time or at least should not decrease.
6. transfer_key.py : Simulates the 3 participants of the collaborative group and an enricher which take part in the key exchange    process. On blockhain, this code flow will be executed through a smart contract.
7. transfer_key_main.py : Testing the transfer_key.py through orchestration of the process.
8. transfer_util.py : As the name suggests, it's a util function for the transfer_key.py.
9. test_enrichment.py : This function mimics one iteration of a single party for training MNIST dataset. 
    It receives global model, then it trains the global model with it's local data
    and then sends the gradients and bias to the Blockchain. This gradients are not "encrypted"
10. test_enrichment2.py : Consider this program as that which will run on the blockchain. Following are the major steps :
    1. Send the encrypted gradients to the enricher.
    2. enricher trains on it's own data and uploads to blockchain.
    3. Rewards enricher based on the accuracy vs compensation plan defined by consensus between parties.
    First part of the program will use test_enrichment to reach a stage where we expect to be in real life before enrichment, i.e. we have encrypted gradients stored on the Blockchain(Here, in our RAM :)). Then, it mimics code flow of the enricher.
11. sigmoid.py : Has a function sigmoid used for neuralnetwork training.
12. enricher.py : Mimics behavior of the enricher. In the paper and real life, we want this entire training to happen in the encrypted domain. However, due to lack of library for encrypted neural network training and testing, we have no choice but to prototype it as enrichment in the decrypted domain.

I have tried to capture the aggregation of the Gradients on the Blockchain in the Hyperledger Fabric Framework.
===> There are 3 files for the blockchain prototype of very basic level of complexity that we wish to achieve.
1. permissions.acl : permissions of the entities. No ACL has been added. Admin has all the access.
2. org.example.mynetwork.cto : This is the model file. It has 1 participant, 3 assets and a smart contract(transaction).
    ==> Participant : Party : Entity that can create assets and invoke transactions.
    ==> Assets : A. Advertisement : This Allows participants to advertise what data they have and what is their business goal if any.
                 B. GradientPerIterationValue : This is an asset which is created by the participant after each iteration of the training. This contains an array of encrypted gradients.
                 C. AggregateGradientsOutput : This is an asset which will be output of the Aggregate function which essentially means the global values of the model after aggregation of individual contribution by the parties.
    ==> Smart Contract : aggregateGradientsTransactionFunction : This function takes into input the GradientPerIterationValue which are submitted by the Participants. Although, it's a dummy function right now which encrypts 2 numbers, add them and decrypt them. After that, it publishes the result as AggregateGradientsOutput. If given values in input, it will add them and then store them as AggregatGradientsOutput.


===> This Blockchain prototype is very trivial compared to what we want to achieve. This is just the demonstration of the fact that our idea works and it is possible to achieve what we want as soon as we have a library which allows encrypted neural network training and testing in the native languages like Python, Java, etc. as well as the languages that Blockchain supports like Solidity for Ethereum or JavaScript for Hyperledger. However, JS is known to have it's problems with Big Integers so it might create a problem.




