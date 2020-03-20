import numpy as np

np.random.seed(10)
downscale_factor = 4
batch_size = 1

image_shape = (320, 240, 3)
shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, image_shape[2])
