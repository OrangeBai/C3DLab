from tensorflow.python.keras.engine.training import Model
from tensorflow.keras.optimizers import Adam, SGD
from models.nets import *
from helper.losses import *
import math


def sr_gan(hr_shape, upscale_num, gen_loss='vgg'):
    if gen_loss == 'vgg':
        loss_fun = vgg_loss(hr_shape)
    elif gen_loss == 'mse':
        loss_fun = 'mse'
    else:
        pass
    downscale_times = int(math.pow(2, upscale_num))
    lr_shape = (hr_shape[0] // downscale_times, hr_shape[1] // downscale_times, hr_shape[2])

    g = generator(lr_shape, upscale_num)
    d = discriminator(hr_shape)

    optimizer = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    g.compile(loss=loss_fun, optimizer=optimizer)
    d.compile(loss="binary_crossentropy", optimizer=optimizer)

    gan_input = Input(shape=lr_shape)

    x = g(gan_input)
    gan_output = d(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[loss_fun, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return g, d, gan
