import os
import numpy as np
import math
import platform


# Input and output configurations
if platform.system() == 'Windows':
    base_path = r'F:\DataSet\fly_vs_fly'
elif platform.system() == 'Linux':
    base_path = r'/scratch/hpc/52/jiangz9/fly_vs_fly'
else:
    base_path = r'F:\DataSet\fly_vs_fly'
agg_video_dir = os.path.join(base_path, 'Aggression')
agg_image_dir = os.path.join(base_path, 'Aggression_Out')
agg_info_path = os.path.join(base_path, 'agg_info.json')

bmb_video_dir = os.path.join(base_path, 'BmB')
bmb_image_dir = os.path.join(base_path, 'BmB_Out')
bmb_info_path = os.path.join(base_path, 'bmb_info.json')

test_dir = os.path.join(base_path, 'test')
weight_dir = os.path.join(base_path, 'weight')

# C3D configurations
# Modifiable variables
clip_shape = (144, 144, 3)


# Super-resolution configurations
# Modifiable variables
upscale_times = 2
batch_size = 2
image_shape_hr = (144, 144, 3)
generator_loss = 'vgg'

# Default variables
downscale = int(math.pow(2, upscale_times))
image_shape_lr = (image_shape_hr[0] // downscale, image_shape_hr[1] // downscale, image_shape_hr[2])
disc_input = (image_shape_hr[1], image_shape_hr[0], 3)


# description
