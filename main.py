from phe import paillier
import json

pub_key,priv_key = paillier.generate_paillier_keypair()

weights4 = 10.1
weights1 = 20.2
weights2 = 30.3
weights3 = 40.4
encrypted = []
tempx = pub_key.encrypt(weights4)
encrypted.append((str(tempx.ciphertext()),tempx.exponent))
tempx2 = paillier.EncryptedNumber(pub_key,int(encrypted[0][0]),int(encrypted[0][1]))

__temp1 = pub_key.encrypt(weights1)
__temp01 = (str(__temp1.ciphertext()),__temp1.exponent)
__temp12 = paillier.EncryptedNumber(pub_key,int(__temp01[0]),int(__temp01[1]))

__temp2 = pub_key.encrypt(weights2)
__temp02 = (str(__temp2.ciphertext()),__temp2.exponent)
__temp22 = paillier.EncryptedNumber(pub_key,int(__temp02[0]),int(__temp02[1]))

__temp3 = pub_key.encrypt(weights3)
__temp03 = (str(__temp3.ciphertext()),__temp3.exponent)
__temp32 = paillier.EncryptedNumber(pub_key,int(__temp03[0]),int(__temp03[1]))





__temp5 = __temp22.__add__(__temp32).__add__(__temp12)
__temp5 = __temp5.__add__(tempx2)

print(priv_key.decrypt(__temp5))

import keyword
print(keyword.kwlist)



# print(priv_key.decrypt(weights))
# enc_with_pub_key = {}
# enc_with_pub_key['public_key'] = { 'g':pub_key.g, 'n':pub_key.n}
# enc_with_pub_key['enc_value'] = (str(weights.ciphertext()),weights.exponent)
# serialised = json.dumps(enc_with_pub_key)
# print(serialised)
# received_dict = json.loads(serialised)
# pk = received_dict['public_key']
# public_key_rec = paillier.PaillierPublicKey(n=int(pk['n']))
# enc_nums_rec = paillier.EncryptedNumber(public_key_rec, int(received_dict['enc_value'][0]), int(received_dict['enc_value'][1]))
# print(priv_key.decrypt(enc_nums_rec))