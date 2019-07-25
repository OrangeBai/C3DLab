import keras
from keras.layers import Input, Conv3D, TimeDistributed, Flatten, Dense
from models.vgg import vgg_base
from models.classifier import *
from config import *
from models.toi_pooling import ToiPoolingConv
from keras.models import Model
import datetime
import os
import pickle
import json



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
        self.model_base_path = model_base_path
        self.model_time = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    def build(self):
        input_tensor = Input(shape=(tube_size, image_size[0], image_size[1], 3))
        rois_input = Input(shape=(self.nb_rois, self.tube_size, 4))
        if self.base_type == 'vgg':
            base = vgg_base
        else:
            base = vgg_base
        x = base(input_tensor, self.block_channels, self.nb_pooling, self.activation, self.kernel_size, self.base_trainable)
        (cls, reg) = rpn(x)
        model_cls = Model(input_tensor, [cls, reg])
        x = ToiPoolingConv(self.pooling_regions, self.nb_rois)([x, rois_input])
        x = classifier(x, channels=self.cls_channels, classes=self.nb_classes, padding=self.cls_padding, activation=self.activation, dense_shape=self.dense_shape, drop_out=self.drop_out)
        model_classifier = Model([input_tensor, rois_input], x)
        return [model_cls, model_classifier]

    def to_pickle(self):
        """
        Save current object to pickle file for further use.
        The save folder is set to be default : ~/Root/Data/
        :param name: File name
        """
        file_folder = os.path.join(self.model_base_path, self.model_time)
        if os.path.isdir(file_folder):
            os.rmdir(file_folder)
        else:
            os.makedirs(file_folder)
        pickle_path = os.path.join(file_folder, self.model_time + '.p')
        with open(pickle_path, 'wb') as f:
            pickle.dump(self, f)
        json_path = os.path.join(file_folder, self.model_time + '.json')
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f)


if __name__ == '__main__':
    model = C3D()
    md = model.build()
    model.to_pickle()
