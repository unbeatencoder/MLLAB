from phe import paillier
from transfer_key_util import getRandom
def transferKey(enc_old_values,pub_key_old,pub_key_new,n):
    getRandom01,getRandom02 = getRandom(n,pub_key_old,pub_key_new)
    getRandom11,getRandom12 = getRandom(n,pub_key_old,pub_key_new)
    getRandom21,getRandom22 = getRandom(n,pub_key_old,pub_key_new)
    getRandom31,getRandom32 = getRandom(n,pub_key_old,pub_key_new)
    for i in range(n):
        __temp = enc_old_values.__add__(getRandom01).__add__(getRandom11).__add__(getRandom21).__add__(getRandom31)
        enc_new_values = __temp.__add__(getRandom02).__add__(getRandom12).__add__(getRandom22).__add__(getRandom32)
    return enc_new_values