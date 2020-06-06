
from phe import paillier
import random
def getRandom(n,pub_key_old,pub_key_new):
    randomold = []
    randomnew = []
    for i in range(n):
        __random0 = random.random()
        __random = pub_key_old.encrypt(__random0)
        __negativeRandom0 = -1* __random0
        __negativeRandom = pub_key_new.encrypt(__negativeRandom0) 
        randomold.append((str(__random.ciphertext()),__random.exponent))
        randomnew.append((str(__negativeRandom.ciphertext()),__negativeRandom.exponent))
    return randomold,randomnew

