##This function handles the transfers of key whenever an enricher is added to the existing collaborative group
# without leaking the gradients.

from phe import paillier
from transfer_key_util import getRandom


def transferKey(enc_old_values,pub_key_old,pub_key_new,n,priv_key_old):
    #emulates the 3 participants of the collaborative group and an enricher which take part in the key exchange process.
    #Each of them provide 2 set of n random values : 1. Random values encrypted with old public key. 
    # 2. Same random values negated and encrypted with the new public key. 
    #getRandomXY represents X === Participant number  
    # Y === if == 1 ==> set 1 described above
    #   === if == 2 ==> set 2 described above  

    getRandom01,getRandom02 = getRandom(n,pub_key_old,pub_key_new)
    getRandom11,getRandom12 = getRandom(n,pub_key_old,pub_key_new)
    getRandom21,getRandom22 = getRandom(n,pub_key_old,pub_key_new)
    getRandom31,getRandom32 = getRandom(n,pub_key_old,pub_key_new)

    enc_new_values = []

    for i in range(n):
        #Recreating numbers from exponent and basis
        __temp0 = paillier.EncryptedNumber(pub_key_old,int(getRandom01[i][0]),int(getRandom01[i][1]))
        __temp1 = paillier.EncryptedNumber(pub_key_old,int(getRandom11[i][0]),int(getRandom11[i][1]))
        __temp2 = paillier.EncryptedNumber(pub_key_old,int(getRandom21[i][0]),int(getRandom21[i][1]))
        __temp3 = paillier.EncryptedNumber(pub_key_old,int(getRandom31[i][0]),int(getRandom31[i][1]))
        __temp10 = paillier.EncryptedNumber(pub_key_new,int(getRandom02[i][0]),int(getRandom02[i][1]))
        __temp11 = paillier.EncryptedNumber(pub_key_new,int(getRandom12[i][0]),int(getRandom12[i][1]))
        __temp12 = paillier.EncryptedNumber(pub_key_new,int(getRandom22[i][0]),int(getRandom22[i][1]))
        __temp13 = paillier.EncryptedNumber(pub_key_new,int(getRandom32[i][0]),int(getRandom32[i][1]))
        __enc_old_values0 = paillier.EncryptedNumber(pub_key_old,int(enc_old_values[i][0]),int(enc_old_values[i][1]))

        #Add random values encrypted with old_encryption_keys to the original numbers.
        __temp = __enc_old_values0.__add__(__temp0).__add__(__temp1).__add__(__temp2).__add__(__temp3)

        #Decrypt values which is summation of old values and random values given by each participant
        __tempdecrypt = priv_key_old.decrypt(__temp)

        #Encrypt this values with a new key.
        __tempnewencrypt = pub_key_new.encrypt(__tempdecrypt)

        #Subtract random values added in first step to gain original values encrypted in newer key.
        __tempnew = __tempnewencrypt.__add__(__temp11).__add__(__temp10).__add__(__temp12).__add__(__temp13)

        #Append and return.
        enc_new_values.append((str(__tempnew.ciphertext()),__tempnew.exponent))
        
    return enc_new_values