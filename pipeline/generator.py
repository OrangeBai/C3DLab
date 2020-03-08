import os
import config
import json
import math
import cv2
import numpy as np


class Generator(object):
    def __init__(self):
        self.data_store = None
        self.clip_length = 5
        self.__load_info__()
        self.gen_label(0, 50)
        print(1)

    def __load_info__(self):
        with open(config.video_info, 'r') as file:
            self.data_store = json.load(file)

    def gen_label(self, video_idx, pic_idx):
        """
        transfer data index into data label
        :param video_idx: video number
        :param pic_idx: frame idx
        :return: pic, label
        """
        cur_clip = []
        cur_tag = {'action': [], 'pos': []}
        cur_video_info = self.data_store[video_idx]
        video_length = 36000
        st_idx = pic_idx - math.ceil((self.clip_length - 1) / 2)
        ed_idx = pic_idx + math.ceil((self.clip_length - 1) / 2)
        if ed_idx >= video_length or st_idx < 0:
            return None
        for i in range(st_idx, ed_idx + 1):
            cur_img_path = os.path.join(cur_video_info['img_dir'], str(i).zfill(6) + '.jpg')
            cur_img = cv2.imread(cur_img_path)
            cur_clip.append(cur_img)

        video_label = cur_video_info['label']
        for i in range(2):
            cur_tag['pos'].append(video_label[1][i][st_idx: ed_idx + 1])
            cur_tag['action'].append(video_label[0][i][pic_idx])
        cur_clip = np.array(cur_clip)
        cur_label = [np.array(cur_tag['pos']), np.array(cur_tag['action'])]

        return cur_clip, cur_label

    def __next__(self):
        return self.next()

    def next(self):
        pass


if __name__ == '__main__':
    g = Generator()
    print(1)
