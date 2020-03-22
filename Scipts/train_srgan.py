from models.SPGAN import *
from pipeline.c3dgenrator import *
from config import *
import tensorflow as tf
from helper.utils import *
from tensorflow.keras.utils import Progbar
import time


tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
        print('Set memery growth False')

data_gen = iter(SRGANGenerator(agg_info_path, image_shape_hr, lr_shape=image_shape_lr, batch_size=batch_size))
gen, dis, gan_model = sr_gan(disc_input, upscale_times)
gen.summary(160)
# g.load_weights(r"F:\DataSet\Aggression_Out\test\w2.h5")

epoch_num = 20
epoch_length = 3000

losses = np.zeros((epoch_length, 4))
best_loss = np.Inf

for epoch in range(epoch_num):
    start_time = time.time()
    progbar = Progbar(epoch_length)
    for iter_num in range(epoch_length):
        imgs_hr, tags_hr, imgs_lr, tags_lr = next(data_gen)

        real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
        fake_data_Y = np.random.random_sample(batch_size) * 0.2

        discriminator.trainable = True

        generated_images_sr = gen.predict(imgs_lr)
        loss = gen.train_on_batch(imgs_lr, imgs_hr)

        print(loss)
        d_loss_real = dis.train_on_batch(imgs_hr, real_data_Y)
        d_loss_fake = dis.train_on_batch(generated_images_sr, fake_data_Y)
        discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        print('discriminator loss:{0}'.format(discriminator_loss))

        gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
        gan_loss = gan_model.train_on_batch(imgs_lr, [imgs_hr, gan_Y])
        print('gan loss:{0}'.format(gan_loss))

        if iter_num % 100 == 0:
            pics = denormalize_m11(generated_images_sr)
            for j in range(batch_size):
                draw_pic(imgs_hr[j], name=str(iter_num // 100) + '_0')
                draw_pic(pics[j], name=str(iter_num // 100) + '_1')

        save_weights(gan_model, '')

print(1)
