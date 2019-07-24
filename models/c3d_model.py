import keras
from keras.layers import Input, Conv3D, TimeDistributed, Flatten, Dense
from models.vgg import vgg_base
from models.classifier import *
from config import *
from models.toi_pooling import ToiPoolingConv
from keras.models import Model


class C3D:
    def __init__(self):
        self.base_type = base_type
        self.image_size = image_size
        self.tube_size = tube_size

        self.block_channels = block_channels
        self.nb_pooling = nb_pooling
        self.activation = bs_activation
        self.kernel_size = bs_kernel_size
        self.base_trainable = base_trainable

        self.pooling_regions = pooling_regions
        self.nb_rois = nb_rois

        self.cls_channels = 16
        self.cls_padding = cls_padding
        self.cls_activation = cls_activation
        self.dense_shape = dense_shape
        self.drop_out = drop_out
        self.nb_classes = nb_classes

        self.model = None

        self.input_tensor = Input(shape=(tube_size, image_size[0], image_size[1], 3))
        self.rois_input = Input(shape=(None, 4 * self.tube_size))

    def build(self):
        if self.base_type == 'vgg':
            base = vgg_base
        else:
            base = vgg_base
        x = base(self.input_tensor, self.block_channels, self.nb_pooling, self.activation, self.kernel_size, self.base_trainable)
        (cls, reg) = rpn(x)
        model_cls = Model(self.input_tensor, [cls, reg])
        model_cls.summary(120)
        x = ToiPoolingConv(self.pooling_regions, self.nb_rois)([x, self.rois_input])
        x = classifier(x, channels=self.cls_channels, classes=self.nb_classes, padding=self.cls_padding, activation=self.activation, dense_shape=self.dense_shape, drop_out=self.drop_out)
        model_classifier = Model([self.input_tensor, self.rois_input], x)
        model_classifier.summary(120)
        return model_classifier


if __name__ == '__main__':
    model = C3D()
    md = model.build()
