import keras
import config
from keras.layers import Input, Conv3D, MaxPooling3D, BatchNormalization, TimeDistributed
from keras.models import Model


def vgg_base(input_tensor=None, block_channels=(8, 16, 32, 32), nb_pooling=4, activation='sigmoid', kernel_size=(1, 3, 3),
             base_trainable=True):
    channel1 = block_channels[0]
    channel2 = block_channels[1]
    channel3 = block_channels[2]
    channel4 = block_channels[3]

    x = BatchNormalization(axis=4)(input_tensor)

    # Block 1
    x = Conv3D(channel1, kernel_size, activation=activation, name='block1_conv1', padding='same', trainable=base_trainable)(x)
    x = Conv3D(channel1, kernel_size, activation=activation, name='block1_conv2', padding='same', trainable=base_trainable)(x)
    x = Conv3D(channel1, kernel_size, activation=activation, name='block1_conv3', padding='same', trainable=base_trainable)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), name='block1_pool1')(x)

    # Block 2
    x = Conv3D(channel2, kernel_size, activation=activation, name='block2_conv1', padding='same', trainable=base_trainable)(x)
    x = Conv3D(channel2, kernel_size, activation=activation, name='block2_conv2', padding='same', trainable=base_trainable)(x)
    x = Conv3D(channel2, kernel_size, activation=activation, name='block2_conv3', padding='same', trainable=base_trainable)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), name='block2_pool1')(x)

    # Block 3
    x = Conv3D(channel3, kernel_size, activation=activation, name='block3_conv1', padding='same', trainable=base_trainable)(x)
    x = Conv3D(channel3, kernel_size, activation=activation, name='block3_conv2', padding='same', trainable=base_trainable)(x)
    x = Conv3D(channel3, kernel_size, activation=activation, name='block3_conv3', padding='same', trainable=base_trainable)(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), name='block3_pool1')(x)

    # Block 4
    x = Conv3D(channel4, kernel_size, activation=activation, name='block4_conv1', padding='same', trainable=True)(x)
    x = Conv3D(channel4, kernel_size, activation=activation, name='block4_conv2', padding='same', trainable=True)(x)
    x = Conv3D(channel4, kernel_size, activation=activation, name='block4_conv3', padding='same', trainable=True)(x)
    if nb_pooling == 4:
        x = MaxPooling3D(pool_size=(1, 2, 2), name='block4_pool1')(x)
    return x


if __name__ == '__main__':
    it = Input(shape=(5, 144, 144, 3), batch_shape=(8, 5, 144, 144, 3), name='input1')
    base = vgg_base(it, nb_pooling=3)
    x = Conv3D(32, kernel_size=(3, 3, 3), padding= 'same')(base)

    model = Model(it, x)
    model.summary()
