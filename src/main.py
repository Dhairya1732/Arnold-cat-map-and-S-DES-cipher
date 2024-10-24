import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from sdes import *
from arnold_cat_map import *

def main():
    # Load and prepare the image
    image_path = 'test.jpeg'  # Path to your image
    original_image = Image.open(image_path).convert('RGB')
    N = min(original_image.size)  # Make the image square
    original_image = original_image.resize((N, N))
    image_array = np.array(original_image)
    
    # Scramble the image using the Arnold Cat Map
    iterations = 30  # Number of iterations to scramble
    scrambled_image = arnold_cat_map(image_array, iterations)

    # Generate keys for S-DES
    key_bits = generate_random_key()
    key1, key2 = key_generation(key_bits)

    # Encrypt the scrambled image
    encrypted_image = encrypt_image(scrambled_image, key1, key2)

    # Decrypt the image
    decrypted_image = decrypt_image(encrypted_image, key1, key2)

    # Unscramble the decrypted image using the inverse Arnold Cat Map
    unscrambled_image = inverse_arnold_cat_map(decrypted_image, iterations)

    # Convert images back to PIL format
    encrypted_image_pil = Image.fromarray(encrypted_image.astype(np.uint8))
    decrypted_image_pil = Image.fromarray(unscrambled_image.astype(np.uint8))

    # Display the original, encrypted, and decrypted images
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(scrambled_image)
    axes[1].set_title("Scrambled Image (Iterations = "+str(iterations)+")")
    axes[1].axis('off') 

    axes[2].imshow(encrypted_image_pil)
    axes[2].set_title("Encrypted Image")
    axes[2].axis('off')
    
    axes[3].imshow(decrypted_image_pil)
    axes[3].set_title("Decrypted Image")
    axes[3].axis('off')
    
    plt.show()

if __name__ == '__main__':
    main()