#This file contains a utility functions for transfer of keys file.

from phe import paillier
import random

#This function returns n random values encrypted with both the public keys old as well as new. But, the value encrypted with the 
# second key will be negative of the first as after change of key, we will need to subtract same random value which was added.
#In the real life implementation, this n will be equivalent to the total number of gradients, bias, etc. values 
# stored on the Blokchain which needs to be tranferred to a new key where enricher is part of the key.
def getRandom(n,pub_key_old,pub_key_new):
    randomold = []
    randomnew = []
    for i in range(n):
        #Generate a random number.
        __random0 = random.random()
        #Encrypt with old key.
        __random = pub_key_old.encrypt(__random0)
        #Negate the random number that is generated.
        __negativeRandom0 = -1* __random0
        #Encrypt with new key
        __negativeRandom = pub_key_new.encrypt(__negativeRandom0) 
        #Append and return.
        randomold.append((str(__random.ciphertext()),__random.exponent))
        randomnew.append((str(__negativeRandom.ciphertext()),__negativeRandom.exponent))
    return randomold,randomnew

