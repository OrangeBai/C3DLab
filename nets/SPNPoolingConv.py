from tensorflow_core.python.keras.engine.base_layer import Layer
import tensorflow as tf


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