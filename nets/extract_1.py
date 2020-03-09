from tensorflow.python.keras.layers.convolutional import Conv2D, Conv3D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.wrappers import TimeDistributed


down_scale = 4
def extract_layer(input_tensor=None):

    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same',
                               name='block1_conv1', trainable=True))(input_tensor)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same',
                               name='block1_conv2', trainable=True))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same',
                               name='block2_conv1', trainable=True))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same',
                               name='block2_conv2', trainable=True))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    return x
