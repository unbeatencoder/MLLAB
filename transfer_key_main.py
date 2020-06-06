from transfer_key import transferKey
from phe import paillier
import random

pub_key_old,priv_key_old = paillier.generate_paillier_keypair()
pub_key_new, priv_key_new = paillier.generate_paillier_keypair()
n = 10
old = []
for i in range(n):
    old.append(random.uniform(0,5))
enc_old = []
for i in range(n):
    __temp = pub_key_old.encrypt(old[i])
    enc_old.append((str(__temp.ciphertext()),__temp.exponent))
print(old)

enc_new = transferKey(enc_old, pub_key_old,pub_key_new,n,priv_key_old)
new = []
for i in range(10):
    __temp = paillier.EncryptedNumber(pub_key_new,int(enc_new[i][0]),int(enc_new[i][1]))
    __decrypted = priv_key_new.decrypt(__temp)
    new.append(__decrypted)

print(new)