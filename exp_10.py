import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# S-DES Tables (Example: You can replace with actual values if necessary)
SBOX1 = [[1, 0, 3, 2], [3, 2, 1, 0], [0, 2, 1, 3], [3, 1, 3, 2]]
SBOX2 = [[0, 1, 2, 3], [2, 0, 1, 3], [3, 0, 1, 0], [2, 1, 0, 3]]

# Arnold Cat Map for Scrambling
def arnold_cat_map(image, iterations, a=1, b=1):
    N = image.shape[0]  # Assuming square image (N x N)
    scrambled_image = np.copy(image)
    
    for _ in range(iterations):
        new_image = np.zeros_like(scrambled_image)
        for i in range(N):
            for j in range(N):
                # Correct formula for Arnold Cat Map
                x_new = (i + (b * j)) % N
                y_new = ((a * i) + ((a * b + 1) * j)) % N
                new_image[x_new, y_new] = scrambled_image[i, j]
        scrambled_image = new_image  
    return scrambled_image


# Logistic Map for Chaotic Sequence
def logistic_map(x0, b, n):
    chaotic_sequence = []
    x = x0
    for _ in range(n):
        x = b * x * (1 - x)
        chaotic_sequence.append(x)
    return chaotic_sequence

# Function to convert chaotic sequence float to binary
def float_to_binary(value, bits=8):
    # Scale the float to an integer (between 0 and 255), then convert to binary
    return format(int(value * (2**bits)), f'0{bits}b')

# S-DES encryption logic with one round (you can modify for multiple rounds)
def sdes_round(binary_pixel, key):
    # XOR the binary pixel with the key
    xor_result = [bit ^ key_bit for bit, key_bit in zip(binary_pixel, key)]

    # S-Box substitution (based on xor_result)
    # Select 2-bit segments from xor_result for S-BOX lookup
    row1 = xor_result[0] * 2 + xor_result[3]
    col1 = xor_result[1] * 2 + xor_result[2]
    sbox_output1 = SBOX1[row1][col1]

    row2 = xor_result[4] * 2 + xor_result[7]
    col2 = xor_result[5] * 2 + xor_result[6]
    sbox_output2 = SBOX2[row2][col2]

    # Combine S-BOX outputs (simplified as 4 bits)
    encrypted_pixel = [int(b) for b in f'{sbox_output1:02b}{sbox_output2:02b}']

    # Return the 8-bit encrypted pixel (we'll expand back to 8 bits)
    return encrypted_pixel + [0, 0, 0, 0]  # Padding the remaining bits

# Encrypt image using S-DES and Logistic Map
def encrypt_image(image, logistic_seq):
    encrypted_image = np.copy(image)
    N = image.shape[0]
    seq_len = len(logistic_seq)
    
    for i in range(N):
        for j in range(N):
            pixel_value = encrypted_image[i, j]
            binary_pixel = format(pixel_value, '08b')
            chaotic_seq = logistic_seq[(i*N + j) % seq_len]
            key = float_to_binary(chaotic_seq)  # Convert chaotic float to 8-bit binary
            # Ensure both pixel and key are 8 bits long and properly mapped
            encrypted_pixel = sdes_round(list(map(int, binary_pixel)), list(map(int, key)))
            encrypted_image[i, j] = int("".join(map(str, encrypted_pixel[:8])), 2)  # Keep only first 8 bits
    
    return encrypted_image

# Decrypt image (same process as encryption for S-DES round in reverse)
def decrypt_image(encrypted_image, logistic_seq):
    decrypted_image = np.copy(encrypted_image)
    N = encrypted_image.shape[0]
    seq_len = len(logistic_seq)
    
    for i in range(N):
        for j in range(N):
            encrypted_pixel_value = decrypted_image[i, j]
            binary_pixel = format(encrypted_pixel_value, '08b')
            chaotic_seq = logistic_seq[(i*N + j) % seq_len]
            key = float_to_binary(chaotic_seq)  # Convert chaotic float to 8-bit binary
            # Decrypt pixel using the same key
            decrypted_pixel = sdes_round(list(map(int, binary_pixel)), list(map(int, key)))
            decrypted_image[i, j] = int("".join(map(str, decrypted_pixel[:8])), 2)  # Keep only first 8 bits
    
    return decrypted_image

# Reverse Arnold Cat Map for Unscrambling
def reverse_arnold_cat_map(image, iterations, a=1, b=1):
    N = image.shape[0]
    unscrambled_image = np.copy(image)
    
    for _ in range(iterations):
        new_image = np.zeros_like(unscrambled_image)
        for i in range(N):
            for j in range(N):
                x_old = (((a * b + 1) * i) - (b * j)) % N
                y_old = ((-a * i) +  j) % N
                new_image[i, j] = unscrambled_image[x_old, y_old]
        unscrambled_image = new_image
    return unscrambled_image

# Function to compare original and decrypted images
def compare_images(scrambled_image, decrypted_image):
    return np.array_equal(scrambled_image, decrypted_image)

# Main Function to execute the algorithm (Modified Part)
def main():
    # Load the image (Convert to grayscale)
    image = Image.open('test.jpeg').convert('L')
    image_np = np.array(image)

    # Step 1: Apply Arnold Cat Map to scramble the image
    scrambled_image = arnold_cat_map(image_np, iterations=30)

    # Step 2: Reverse Arnold Cat Map to check if unscrambling works correctly
    unscrambled_test_image = reverse_arnold_cat_map(scrambled_image, iterations=30)

    # Compare original image with the unscrambled test image
    if compare_images(image_np, unscrambled_test_image):
        print("Scrambling and unscrambling successful: Arnold Cat Map works correctly.")
    else:
        print("Scrambling and unscrambling failed: Check Arnold Cat Map implementation.")

    # Step 3: Generate Logistic Map chaotic sequence
    logistic_seq = logistic_map(0.5, 3.9, 256*256)

    # Step 4: Encrypt the image using S-DES and Logistic Map keys
    encrypted_image = encrypt_image(scrambled_image, logistic_seq)

    # Step 5: Decrypt the image using S-DES
    decrypted_image = decrypt_image(encrypted_image, logistic_seq)

    # Step 6: Reverse Arnold Cat Map to unscramble the decrypted image
    unscrambled_image = reverse_arnold_cat_map(decrypted_image, iterations=30)

    # Step 7: Compare original image with unscrambled decrypted image
    if compare_images(image_np, unscrambled_image):
        print("Decryption successful: Decrypted image matches the original!")
    else:
        print("Decryption failed: Decrypted image does not match the original.")

    # Step 8: Display original, encrypted, and unscrambled decrypted images
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image_np, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Encrypted Image')
    plt.imshow(encrypted_image, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Unscrambled Decrypted Image')
    plt.imshow(unscrambled_image, cmap='gray')

    plt.show()

if __name__ == "__main__":
    main()