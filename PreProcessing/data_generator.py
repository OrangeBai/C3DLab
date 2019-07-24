from PreProcessing import fly_tag as ft
import config
import cv2
import os
import numpy as np
import random


class DataGenerator:
    def __init__(self):
        self.config = config

        self.base_path = config.output_path
        self.pos = None
        self.actions = None
        self.clip_length = config.clip_length

        self.__load_pickle__()

    def __load_pickle__(self):
        pickle_info = ft.load_tags_and_info(self.base_path)
        self.pos = pickle_info[0]
        self.actions = pickle_info[1]

    def generate_box_with_tag(self, frame_info):
        """
        Get bounding boxes of a particular frame
        :param frame_info: tuple: (data_base, frame_num)
        :return: bounding boxes and tags
        """
        action_0 = ft.check_action(self.actions, (frame_info[0], 0, frame_info[2]))
        action_1 = ft.check_action(self.actions, (frame_info[0], 1, frame_info[2]))
        pos_0 = self.pos[frame_info[0]][0, frame_info[1], :2]
        pos_1 = self.pos[frame_info[0]][1, frame_info[1], :2]

        return action_0, action_1, pos_0, pos_1

    def emit_data(self, frame_info):
        img = []
        base_path = frame_info[0]
        frame_num = frame_info[2]
        if frame_num < 5 or frame_num > 53995:
            return None
        for num in range(frame_num - int(config.clip_length / 2), frame_num + int(config.clip_length / 2) + 1):
            cur_img = cv2.imread(os.path.join(base_path, str(num).zfill(6) + '.jpg'))
            img.append(cur_img)
        img = np.array(img)
        return img, self.generate_box_with_tag(frame_info)

    def generator(self):
        while True:
            action_num = random.randint(0, 6)
            action_name = self.actions[1][action_num]
            action_list = self.actions[0][action_name]
            data_num = random.randint(0, len(action_list) - 1)
            data_info = action_list[data_num]
            yield self.emit_data(data_info)


if __name__ == '__main__':
    print(1)
    GT = DataGenerator()
    gen = GT.generator()
    print(next(gen))
    print(next(gen))

#
# rec = [[int(pos_0[0] - 6), int(pos_0[1] - 2), int(pos_0[0] + 6), int(pos_0[1] + 2)]]
# image_path = os.path.join(frame_info[0], str(frame_info[1]).zfill(self.config.name_length) + '.' + 'jpg')
# image = cv2.imread(image_path)
# image_helper.draw_rec(image, test_path=r'F:\DataSet\Aggression_Out\Test', boxes=rec)
