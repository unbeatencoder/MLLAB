from phe import paillier
from transfer_key_util import getRandom
def transferKey(enc_old_values,pub_key_old,pub_key_new,n,priv_key_old):
    getRandom01,getRandom02 = getRandom(n,pub_key_old,pub_key_new)
    getRandom11,getRandom12 = getRandom(n,pub_key_old,pub_key_new)
    getRandom21,getRandom22 = getRandom(n,pub_key_old,pub_key_new)
    getRandom31,getRandom32 = getRandom(n,pub_key_old,pub_key_new)
    enc_new_values = []
    for i in range(n):
        __temp0 = paillier.EncryptedNumber(pub_key_old,int(getRandom01[i][0]),int(getRandom01[i][1]))
        __temp1 = paillier.EncryptedNumber(pub_key_old,int(getRandom11[i][0]),int(getRandom11[i][1]))
        __temp2 = paillier.EncryptedNumber(pub_key_old,int(getRandom21[i][0]),int(getRandom21[i][1]))
        __temp3 = paillier.EncryptedNumber(pub_key_old,int(getRandom31[i][0]),int(getRandom31[i][1]))
        __temp10 = paillier.EncryptedNumber(pub_key_new,int(getRandom02[i][0]),int(getRandom02[i][1]))
        __temp11 = paillier.EncryptedNumber(pub_key_new,int(getRandom12[i][0]),int(getRandom12[i][1]))
        __temp12 = paillier.EncryptedNumber(pub_key_new,int(getRandom22[i][0]),int(getRandom22[i][1]))
        __temp13 = paillier.EncryptedNumber(pub_key_new,int(getRandom32[i][0]),int(getRandom32[i][1]))
        __enc_old_values0 = paillier.EncryptedNumber(pub_key_old,int(enc_old_values[i][0]),int(enc_old_values[i][1]))
        __temp = __enc_old_values0.__add__(__temp0).__add__(__temp1).__add__(__temp2).__add__(__temp3)
        __tempdecrypt = priv_key_old.decrypt(__temp)
        print(__tempdecrypt)
        __tempnewencrypt = pub_key_new.encrypt(__tempdecrypt)
        __tempnew = __tempnewencrypt.__add__(__temp11).__add__(__temp10).__add__(__temp12).__add__(__temp13)
        enc_new_values.append((str(__tempnew.ciphertext()),__tempnew.exponent))
        
    return enc_new_values