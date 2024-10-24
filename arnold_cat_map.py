import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def arnold_cat_map(image, iterations):
    scrambled_image = np.array(image)
    N = scrambled_image.shape[0]
    
    for _ in range(iterations):
        new_image = np.zeros_like(scrambled_image)
        for x in range(N):
            for y in range(N):
                new_x = (x + y) % N
                new_y = (x + 2 * y) % N
                new_image[new_x, new_y] = scrambled_image[x, y]
        scrambled_image = new_image
    return scrambled_image

def inverse_arnold_cat_map(image, iterations):
    unscrambled_image = np.array(image)
    N = unscrambled_image.shape[0]
    
    for _ in range(iterations):
        new_image = np.zeros_like(unscrambled_image)
        for x in range(N):
            for y in range(N):
                new_x = (2 * x - y) % N
                new_y = (-x + y) % N
                new_image[new_x, new_y] = unscrambled_image[x, y]
        unscrambled_image = new_image
    return unscrambled_image

# Load the image and resize it to a square if necessary
image_path = 'test.jpeg'  # Path to your image
original_image = Image.open(image_path).convert('RGB')
N = min(original_image.size)  # Make the image square
original_image = original_image.resize((N, N))

# Convert the image to a NumPy array
image_array = np.array(original_image)

# Apply the Arnold Cat Map for scrambling
iterations = 30  # Number of iterations to scramble
scrambled_image = arnold_cat_map(image_array, iterations)

# Apply the inverse Arnold Cat Map for unscrambling
unscrambled_image = inverse_arnold_cat_map(scrambled_image, iterations)

# Convert the scrambled and unscrambled images back to PIL format
scrambled_image_pil = Image.fromarray(scrambled_image)
unscrambled_image_pil = Image.fromarray(unscrambled_image)

# Display the original, scrambled, and unscrambled images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(scrambled_image_pil)
axes[1].set_title(f"Scrambled Image\n(Iterations: {iterations})")
axes[1].axis('off')

axes[2].imshow(unscrambled_image_pil)
axes[2].set_title("Unscrambled Image")
axes[2].axis('off')

plt.show()