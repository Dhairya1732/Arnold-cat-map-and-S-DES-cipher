import numpy as np
import random

# --- S-DES helper functions ---
def generate_random_key():
    return [random.randint(0, 1) for _ in range(10)]

def permute(bits, permutation):
    return [bits[i - 1] for i in permutation]

def shift_left(bits, shifts):
    return bits[shifts:] + bits[:shifts]

def key_generation(key):
    P10 = [3, 5, 2, 7, 4, 10, 1, 9, 8, 6]
    P8 = [6, 3, 7, 4, 8, 5, 10, 9]
    
    key = permute(key, P10)
    left, right = key[:5], key[5:]
    left, right = shift_left(left, 1), shift_left(right, 1)
    key1 = permute(left + right, P8)
    
    left, right = shift_left(left, 2), shift_left(right, 2)
    key2 = permute(left + right, P8)
    
    return key1, key2

def xor(bits1, bits2):
    return [b1 ^ b2 for b1, b2 in zip(bits1, bits2)]

def sbox(bits, sbox):
    row = bits[0] * 2 + bits[3]
    col = bits[1] * 2 + bits[2]
    return [int(x) for x in f"{sbox[row][col]:02b}"]

def f_function(right, key):
    EP = [4, 1, 2, 3, 2, 3, 4, 1]
    P4 = [2, 4, 3, 1]
    
    s0 = [[1, 0, 3, 2],
          [3, 2, 1, 0],
          [0, 2, 1, 3],
          [3, 1, 3, 2]]
    
    s1 = [[0, 1, 2, 3],
          [2, 0, 1, 3],
          [3, 0, 1, 0],
          [2, 1, 0, 3]]
    
    right_expanded = permute(right, EP)
    xor_output = xor(right_expanded, key)
    
    left_xor, right_xor = xor_output[:4], xor_output[4:]
    left_sbox = sbox(left_xor, s0)
    right_sbox = sbox(right_xor, s1)
    
    return permute(left_sbox + right_sbox, P4)

def sdes_encrypt_block(bits, key1, key2):
    IP = [2, 6, 3, 1, 4, 8, 5, 7]
    IP_inv = [4, 1, 3, 5, 7, 2, 8, 6]
    
    bits = permute(bits, IP)
    
    left, right = bits[:4], bits[4:]
    left = xor(left, f_function(right, key1))
    
    left, right = right, left
    left = xor(left, f_function(right, key2))
    
    return permute(left + right, IP_inv)

def sdes_decrypt_block(bits, key1, key2):
    return sdes_encrypt_block(bits, key2, key1)

def int_to_bits(n, bit_size=8):
    return [int(x) for x in format(n, f'0{bit_size}b')]

def bits_to_int(bits):
    return int(''.join(map(str, bits)), 2)

def encrypt_image(image_array, key1, key2):
    h, w, _ = image_array.shape
    encrypted_array = np.copy(image_array)
    
    for i in range(h):
        for j in range(w):
            # Encrypt each color channel
            for k in range(3):
                encrypted_array[i, j, k] = bits_to_int(sdes_encrypt_block(int_to_bits(image_array[i, j, k]), key1, key2))

    return encrypted_array

def decrypt_image(image_array, key1, key2):
    h, w, _ = image_array.shape
    decrypted_array = np.copy(image_array)
    
    for i in range(h):
        for j in range(w):
            # Decrypt each color channel
            for k in range(3):
                decrypted_array[i, j, k] = bits_to_int(sdes_decrypt_block(int_to_bits(decrypted_array[i, j, k]), key1, key2))

    return decrypted_array