import keras
from keras.layers import Dense, TimeDistributed, Flatten, Conv3D, Dropout


def rpn(base_layers):
    x = Conv3D(32, (1, 3, 3), padding='same', activation='sigmoid', kernel_initializer='normal', name='rpn_conv1')(base_layers)

    x_class = Conv3D(1, (5, 1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_reg = Conv3D(4, (5, 1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_reg]


def classifier(base_layers, classes, channels, padding, activation, dense_shape, drop_out):
    x = TimeDistributed(Conv3D(channels, (3, 3, 3), padding=padding, activation=activation))(base_layers)
    x = TimeDistributed(Conv3D(channels, (3, 3, 3), padding=padding, activation=activation))(x)
    x = TimeDistributed(Conv3D(channels, (3, 3, 3), padding=padding, activation=activation))(x)
    x = TimeDistributed(Flatten(name='flatten'))(x)

    x = TimeDistributed(Dense(dense_shape[0], activation='sigmoid'))(x)
    x = TimeDistributed(Dropout(drop_out))(x)

    x = TimeDistributed(Dense(dense_shape[1], activation='sigmoid'))(x)
    x = TimeDistributed(Dropout(drop_out))(x)

    x = TimeDistributed(Dense(dense_shape[2], activation='sigmoid'))(x)
    x = TimeDistributed(Dropout(drop_out))(x)

    x = TimeDistributed(Dense(classes))(x)

    return x

