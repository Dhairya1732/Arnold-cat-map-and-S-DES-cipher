import numpy as np

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