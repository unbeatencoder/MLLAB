
from phe import paillier
import random
def getRandom(n,pub_key_old,pub_key_new):
    randomold = []
    randomnew = []
    for i in range(n):
        __random = random.uniform(0.30, 10.12)
        __negativeRandom = -1* __random
        randomold.append(pub_key_old.encrypt(__random))
        randomnew.append(pub_key_new.encrypt(__negativeRandom))
    return randomold,randomnew