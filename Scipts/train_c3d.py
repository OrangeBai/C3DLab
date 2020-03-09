from pipeline.generator import *
from tensorflow.keras.layers import Input
from nets.extract_1 import *
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers.wrappers import TimeDistributed
from nets.RoiPoolingConv import *
from pipeline.SPN import *


num_frames = 5
gen = Generator()

input_tensor = Input((5, 144, 144, 3))
pre_roi = extract_layer(input_tensor)
x_class = TimeDistributed(Conv2D(20, (1, 1), activation='sigmoid', kernel_initializer='uniform',
                                 name='rpn_out_class'))(pre_roi)
x_regr = TimeDistributed(Conv2D(80, (1, 1), activation='linear', kernel_initializer='zero',
                                name='rpn_out_regress'))(pre_roi)

spn_helper = SPN(7, (144, 144))

m = Model([input_tensor], [x_class, x_regr])
m.compile(optimizer='sgd', loss='mae')
m.summary(160)
for i in range(20):
    img, label = gen.next()
    spn_cls = []
    spn_reg = []
    for j in range(num_frames):
        cur_spn_cls, cur_spn_reg, cur_box_cls, cur_box_raw, cur_spn_cls_valid = spn_helper.cal_gt_tags(label[0][j])
        spn_cls.append(cur_spn_cls)
        spn_reg.append(cur_spn_reg)
    spn_cls = np.expand_dims(np.array(spn_cls), 0)
    spn_reg = np.expand_dims(np.array(spn_reg), 0)
    loss = m.train_on_batch(img, [spn_cls, spn_reg])
    print(loss)

