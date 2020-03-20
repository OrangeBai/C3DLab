from nets.SRGAN import *
from helper.losses import *
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.engine.training import Model
from tensorflow.keras.optimizers import Adam
from pipeline.c3dgenrator import *
import tensorflow as tf

tf.config.experimental_run_functions_eagerly(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, False)
        print('Set memery growth False')

np.random.seed(10)
downscale_factor = 4
image_shape = (320, 240, 3)
batch_size = 1


# Combined network
def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, image_shape[2])

generator = Generator(shape).generator()
discriminator = Discriminator(image_shape).discriminator()

adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
generator.compile(loss=vgg_loss_d(image_shape), optimizer=adam)
discriminator.compile(loss="binary_crossentropy", optimizer=adam)

shape = (image_shape[0] // downscale_factor, image_shape[1] // downscale_factor, 3)
gan = get_gan_network(discriminator, shape, generator, adam, vgg_loss_d(image_shape))


srgan_g = SRGANGenerator(config.hr_video_info)
for i in range(1000):
    img, tag = srgan_g.next()
    img = cv2.resize(img, (image_shape[1], image_shape[0]))
    img_lr = cv2.resize(img, (shape[1], shape[0]))
    img = np.expand_dims(img, axis=0).astype('float32')
    img_lr = np.expand_dims(img_lr, axis=0).astype('float32')

    real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
    fake_data_Y = np.random.random_sample(batch_size) * 0.2

    discriminator.trainable = True

    generated_images_sr = generator.predict(img_lr)

    d_loss_real = discriminator.train_on_batch(img, real_data_Y)
    d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
    discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
    print('discriminator loss:{0}'.format(discriminator_loss))

    gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
    gan_loss = gan.train_on_batch(img_lr, [img, gan_Y])
    print('gan loss:{0}'.format(gan_loss))