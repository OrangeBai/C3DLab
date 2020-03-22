from tensorflow.python.keras.layers.convolutional import Conv2D, Conv3D, Conv2DTranspose, UpSampling2D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from tensorflow.python.keras.layers import Dense, Activation, Input, Flatten, add, Lambda
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU
from tensorflow_core.python.keras.engine.base_layer import Layer
import tensorflow_core as tf
from helper.utils import *


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


def res_block_gen(model, kernel_size, filters, strides):
    gen = model

    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(model)
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)

    model = add([gen, model])

    return model


def up_sampling_block(model, kernal_size, filters, strides):
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    # model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters=filters, kernel_size=kernal_size, strides=strides, padding="same")(model)
    model = UpSampling2D(size=2)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


def discriminator_block(model, filters, kernel_size, strides):
    model = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model


def generator(input_shape, upscale_times=2):

    gen_input = Input(shape=input_shape)

    model = Conv2D(filters=64, kernel_size=9, strides=1, padding="same")(gen_input)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(
        model)

    gen_model = model

    # Using 16 Residual Blocks
    for index in range(8):
        model = res_block_gen(model, 3, 64, 1)

    model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = add([gen_model, model])

    # Using 2 UpSampling Blocks
    for index in range(upscale_times):
        model = up_sampling_block(model, 3, 256, 1)

    model = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(model)
    model = Activation('tanh')(model)

    generator_model = Model(inputs=gen_input, outputs=model)

    return generator_model


def discriminator(image_shape):
    dis_input = Input(shape=image_shape)

    model = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(dis_input)
    model = LeakyReLU(alpha=0.2)(model)

    model = discriminator_block(model, 64, 3, 2)
    model = discriminator_block(model, 128, 3, 1)
    model = discriminator_block(model, 128, 3, 2)
    model = discriminator_block(model, 256, 3, 1)
    model = discriminator_block(model, 256, 3, 2)
    model = discriminator_block(model, 512, 3, 1)
    model = discriminator_block(model, 512, 3, 2)

    model = Flatten()(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Dense(1)(model)
    model = Activation('sigmoid')(model)

    discriminator_model = Model(inputs=dis_input, outputs=model)

    return discriminator_model


class SPNPoolingConv(Layer):
    def __init__(self, deviation_ratio, num_frame, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.deviation_ratio = deviation_ratio
        self.num_rois = num_rois
        self.nub_channels = 0
        self.num_frame = num_frame

        super(SPNPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        The input should
        :param input_shape:
        :return:
        """
        self.nub_channels = input_shape[0][4]

    def compute_output_shape(self, input_shape):
        return None, self.num_frame, self.num_rois, self.pool_size, self.pool_size, self.nub_channels

    def call(self, inputs, **kwargs):
        """
        concate SPN layers and
        :param inputs: Input tensors [feature map of resized pic, spn_cls, spn_reg]
        :param kwargs:
        :return:
        """


class RoiPoolingConv(Layer):
    """
    ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networtfs for Visual Recognition,
    tf. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    """

    def __init__(self, pool_size, num_rois, **tfwargs):

        self.pool_size = pool_size
        self.num_rois = num_rois
        self.nb_channels = 0

        super(RoiPoolingConv, self).__init__(**tfwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):

        assert (len(x) == 2)

        img = x[0]
        rois = x[1]

        outputs = []

        for roi_idx in range(self.num_rois):

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = tf.cast(x, 'int32')
            y = tf.cast(y, 'int32')
            w = tf.cast(w, 'int32')
            h = tf.cast(h, 'int32')

            rs = tf.image.resize(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = tf.concat(outputs, axis=0)
        final_output = tf.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        return final_output



#
# class ToiPoolingConv(Layer):
#     def __init__(self, pool_size, num_rois, **kwargs):
#         self.pool_size = pool_size
#         self.nb_rois = num_rois
#         self.nb_frames = None
#         self.nb_channels = None
#
#         super(ToiPoolingConv, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.nb_channels = input_shape[0][4]
#         self.nb_frames = input_shape[0][1]
#
#     def compute_output_shape(self, input_shape):
#         return None, self.nb_rois, self.nb_frames, self.pool_size, self.pool_size, self.nb_channels
#
#     def call(self, x, mask=None):
#         assert (len(x) == 2)
#
#         img = x[0]  # Shape of imgs is (1, nb_frames, rows, cols, channels)
#         rois = x[1]  # shape of rois is (1, nb_rois, nb_frames, 4)
#
#         outputs = []
#         for rois_idx in range(self.nb_rois):
#             output = []
#             for frame_idx in range(self.nb_frames):
#                 x = rois[0, rois_idx, frame_idx, 0]
#                 y = rois[0, rois_idx, frame_idx, 1]
#                 w = rois[0, rois_idx, frame_idx, 2]
#                 h = rois[0, rois_idx, frame_idx, 3]
#
#                 x = K.cast(x, 'int32')
#                 y = K.cast(y, 'int32')
#                 w = K.cast(w, 'int32')
#                 h = K.cast(h, 'int32')
#
#                 rs = tf.image.resize_images(img[0, frame_idx, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
#                 rs = K.expand_dims(rs, axis=0)
#                 output.append(rs)
#
#             output = K.concatenate(output, axis=0)
#             output = K.expand_dims(output, axis=0)
#             outputs.append(output)
#
#         final_output = K.concatenate(outputs, axis=0)
#         final_output = K.expand_dims(final_output, axis=0)
#
#         return final_output
#
#     def get_config(self):
#         config = {'pool_size': self.pool_size,
#                   'nb_rois': self.nb_rois,
#                   'nb_frames': self.nb_frames,
#                   'nb_channels': self.nb_channels}
#         base_config = super(ToiPoolingConv, self).get_config()
#         new_list = list(base_config.items()) + list(config.items())
#
#         return {key: value for (key, value) in new_list}
