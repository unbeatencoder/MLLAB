#This function tries to emulate logic for the transfer of keys whenever an enricher is added to the existing collaborative group
# without leaking the gradients.
from transfer_key import transferKey
from phe import paillier
import random


#Generate 2 keys. pub_key_old and priv_key_old represent the keypair of the existing collaborative group.
# pub_key_new and priv_key_new represent the new keypair which is shared between existing collaborative group as well as the enricher. 
pub_key_old,priv_key_old = paillier.generate_paillier_keypair()
pub_key_new, priv_key_new = paillier.generate_paillier_keypair()

#Generate 10 random integers which stand as the gradients that needs to be transferred from one key to another.
n = 10
old = []
for i in range(n):
    old.append(random.uniform(0,5))


#Encrypt the random integers and this represents the values on the Blockchain by the collaborative group.
enc_old = []
for i in range(n):
    __temp_encrypted_value = pub_key_old.encrypt(old[i])
    enc_old.append((str(__temp_encrypted_value.ciphertext()),__temp_encrypted_value.exponent))


#Transfer of keys happen using the function transferKey which takes following arguments:
#@param enc_old : set of encrypted values whose key will change.
#@param pub_key_old : old public key with which data is already encrypted.
#@param pub_key_new : new public key with which data will be encrypted.
#@param priv_key_old : In real life implementation, this will not be shared with this function. Instead, 
#                      values will be decrypted using the decryption shares received from collaborative party
#                      after adding random values taken from the participants as stimulated in the transferKey function. 
enc_new = transferKey(enc_old, pub_key_old,pub_key_new,n,priv_key_old)

#Decrypting the values after transfer of keys to check if the values are correct indeed.
new = []
for i in range(n):
    __temp_decrypted_value = paillier.EncryptedNumber(pub_key_new,int(enc_new[i][0]),int(enc_new[i][1]))
    __decrypted = priv_key_new.decrypt(__temp_decrypted_value)
    new.append(__decrypted)

#This values should match the output of original random values printed in the old array.
for i in range(n):
    print(old[i],new[i])
