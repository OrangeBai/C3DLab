from helper.losses import *
import cv2
import numpy as np

vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(144, 144, 3))
model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)

loss_cls = VGG_LOSS((144, 144, 3))
loss = loss_cls.vgg_loss

img1_path = r"F:\DataSet\Aggression_Out\movie2\000002.jpg"
img2_path = r"F:\DataSet\Aggression_Out\movie2\000095.jpg"

img1 = np.expand_dims(cv2.imread(img1_path), axis=0).astype('float32')
img2 = np.expand_dims(cv2.imread(img2_path), axis=0).astype('float32')

print(loss(img1, img2))
