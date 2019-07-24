import math

# Pre_processing Info
input_path = r'F:\DataSet\Aggression'
output_path = r'F:\DataSet\Aggression_Out'
action_path = r'F:\DataSet\flyvideo'

# Pickle Settings
pos_pickle_path = r'F:\DataSet\Aggression_Out'
only_pos = False

# Other Config
create_folder_for_video = False
action_names = []

# Video Config
image_format = 'jpg'
name_length = 6
verbose = True
bar_length = 80

# Base Layer config
base_type = 'vgg'
image_size = (144, 144)  # Image Size
block_channels = (8, 16, 32, 32)
tube_size = 5  # Tube Size
nb_pooling = 3
bs_activation = 'sigmoid'
bs_kernel_size = (1, 3, 3)
base_trainable = True  # Base layers are trainable

# ROI config
pooling_regions = 5  # Output size of RoI
nb_rois = 8  # Number of RoIs per batch
RoI_trainable = True  # RoI trainable

# Classifier config
cls_channels = 16   # number of channels for C3D layers
cls_padding = 'same'
cls_activation = 'sigmoid'
dense_shape = [1024, 512, 64]
drop_out = 0.5  # Drop out rate of dense layers
parent = False  # Contains parent classifiers.
nb_classes = 6  # Action classes


# Anchor config
anchor_box_scales = [48, 72, 96]  # Side length of anchor box
anchor_box_ratios = [[1, 1], [1. / math.sqrt(2), 2. / math.sqrt(2)],
                          [2. / math.sqrt(2), 1. / math.sqrt(2)]]  # Ratio of anchor box
num_anchors = len(anchor_box_scales) * len(anchor_box_ratios)  # Number of Anchor boxes

# Data generation config
flip_x = True
flip_y = True

#
clip_length = 5