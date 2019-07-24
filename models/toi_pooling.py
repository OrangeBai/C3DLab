from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class ToiPoolingConv(Layer):
    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.nb_rois = num_rois
        self.nb_frames = None
        self.nb_channels = None

        super(ToiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][4]
        self.nb_frames = input_shape[0][1]

    def compute_output_shape(self, input_shape):
        return None, self.nb_rois, self.nb_frames, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None):
        assert (len(x) == 2)

        img = x[0]  # Shape of imgs is (1, nb_frames, rows, cols, channels)
        rois = x[1]  # shape of rois is (1, nb_rois, 4 * nb_frames)

        outputs = []
        for rois_idx in range(self.nb_rois):
            output = []
            for frame_idx in range(self.nb_frames):
                x = rois[frame_idx, rois_idx, 0]
                y = rois[frame_idx, rois_idx, 1]
                w = rois[frame_idx, rois_idx, 2]
                h = rois[frame_idx, rois_idx, 3]

                x = K.cast(x, 'int32')
                y = K.cast(y, 'int32')
                w = K.cast(w, 'int32')
                h = K.cast(h, 'int32')

                rs = tf.image.resize_images(img[:, frame_idx, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
                rs = K.expand_dims(rs, axis=1)
                output.append(rs)

            output = K.concatenate(output, axis=1)
            output = K.expand_dims(output, axis=1)
            outputs.append(output)

        final_output = K.concatenate(outputs, axis=1)

        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size,
                  'nb_rois': self.nb_rois,
                  'nb_frames': self.nb_frames,
                  'nb_channels': self.nb_channels}
        base_config = super(ToiPoolingConv, self).get_config()
        new_list = list(base_config.items()) + list(config.items())

        return {key: value for (key, value) in new_list}
