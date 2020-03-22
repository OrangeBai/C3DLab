import os
import json
import math
import cv2
import numpy as np
from copy import *
from random import randint, choice
import config as c
from helper.utils import draw_pic
from helper.utils import *


class DataGenerator:
    def __init__(self, info_path, shape=None, batch_size=1):
        self.data_store = None
        self.__load_info__(info_path)
        self.batch_size = batch_size
        self.shape = shape

    def __load_info__(self, info_path):
        with open(info_path, 'r') as file:
            self.data_store = json.load(file)

    def __next__(self):
        pass

    def next(self):
        """
        Next image with tags
        :return:
        """
        pass

    def __iter__(self):
        return self

    @staticmethod
    def np_pack(imgs, tags):
        imgs = np.array(imgs).astype('float32')
        tags[0] = np.array(tags[0]).astype('float32')
        tags[1] = np.array(tags[1]).astype('float32')
        return imgs, tags

    def gen_label(self, clip_idx):
        video_idx, pic_idx = clip_idx
        cur_tag = [[], []]
        cur_video_info = self.data_store[video_idx]
        cur_img_path = os.path.join(cur_video_info['img_dir'], str(pic_idx).zfill(6) + '.jpg')
        cur_img = cv2.imread(cur_img_path)
        video_label = cur_video_info['label']
        for i in range(2):
            cur_tag[0].append(video_label[1][i][pic_idx])
            cur_tag[1].append(video_label[0][i][pic_idx])
        cur_img = np.array(cur_img).astype(float)
        cur_tag[0] = np.array(cur_tag[0]).astype(float)
        cur_tag[1] = np.array(cur_tag[1]).astype(float)
        if self.shape is not None:
            cur_img, cur_tag = self.resize(cur_img, cur_tag, self.shape)
        return cur_img, cur_tag

    @staticmethod
    def resize(img, tag, shape):
        new_tag = deepcopy(tag)
        img_shape = [img.shape[1], img.shape[0], img.shape[2]]

        new_img = cv2.resize(img, shape[:2])
        new_tag[0][:, [0, 2]] = new_tag[0][:, [0, 2]] * shape[0] / img_shape[0]
        new_tag[0][:, [1, 3]] = new_tag[0][:, [1, 3]] * shape[1] / img_shape[1]
        return new_img, new_tag


class C3DGenerator(DataGenerator):
    def __init__(self, info_path, shape=None, batch_size=1):
        super().__init__(info_path, shape, batch_size)
        self.clip_length = 5
        self.action_counter = []
        self.action_indexer = {}
        self.unzip_label()

    def unzip_label(self):
        action_indexer = {}
        for i in range(len(self.data_store)):
            video_info = self.data_store[i]
            labels = video_info['label']
            video_length = video_info['length']
            for j in range(video_length):
                for k in range(2):
                    cur_action = int(labels[0][k][j])
                    if cur_action not in list(action_indexer.keys()):
                        action_indexer[cur_action] = [(i, j)]
                    else:
                        action_indexer[cur_action].append((i, j))
        self.action_indexer = action_indexer
        self.action_counter = [0] * len(action_indexer.keys())
        return

    def gen_label(self, clip_idx):
        """
        transfer data index into data label
        :param clip_idx: (video_idx, frame_idx)
        :return: pic, label
        """
        clip = []
        tag = [[], []]

        video_idx, pic_idx = clip_idx
        st_idx = pic_idx - math.ceil((self.clip_length - 1) / 2)
        ed_idx = pic_idx + math.ceil((self.clip_length - 1) / 2)

        cur_video_info = self.data_store[video_idx]
        video_length = len(cur_video_info['label'][0][0])
        if ed_idx >= video_length or st_idx < 0:
            return None
        for frame_idx in range(st_idx, ed_idx + 1):
            cur_img, cur_tag = super().gen_label(clip_idx)
            clip.append(cur_img)
            tag[0].append(cur_tag[0])
            if frame_idx == pic_idx:
                tag[1] = cur_tag[1]
        return clip, tag

    def __next__(self):
        imgs = []
        tags = [[], []]
        for i in range(self.batch_size):
            img, tag = self.next()
            imgs.append(img)
            tags[0].append(tag[0])
            tags[1].append(tag[1])
        imgs, tags = self.np_pack(imgs, tags)
        return imgs, tags

    def next(self):
        while 1:
            next_action = self.__check_min__()
            frame = choice(self.action_indexer[next_action])
            res = self.gen_label(frame)
            if res is not None:
                break
        self.action_counter[next_action] = self.action_counter[next_action] + 1
        return res

    def __check_min__(self):
        min_num = min(self.action_counter)
        max_num = max(self.action_counter)
        if min_num < max_num * 0.5:
            return self.action_counter.index(min_num)
        else:
            return randint(0, len(self.action_counter) - 1)


class SRGANGenerator(DataGenerator):
    def __init__(self, info_path, new_shape, lr_shape, batch_size=1):
        super().__init__(info_path, new_shape, batch_size)
        self.lr_shape = lr_shape

    def gen_label(self, clip_idx):
        img_hr, tag_hr = super().gen_label(clip_idx)
        img_lr, tag_lr = self.resize(img_hr, tag_hr, self.lr_shape)
        return img_hr, tag_hr, img_lr, tag_lr

    def __next__(self):
        imgs_hr, imgs_lr = [], []
        tags_hr, tags_lr = [[], []], [[], []]
        for i in range(self.batch_size):
            img_hr, tag_hr, img_lr, tag_lr = self.next()

            imgs_hr.append(img_hr)
            tags_hr[0].append(tag_hr[0])
            tags_hr[1].append(tag_hr[1])

            imgs_lr.append(img_lr)
            tags_lr[0].append(tag_lr[0])
            tags_lr[1].append(tag_lr[1])
        imgs_hr, tags_hr = self.np_pack(imgs_hr, tags_hr)
        imgs_lr, tags_lr = self.np_pack(imgs_lr, tags_lr)
        imgs_hr = normalize_m11(imgs_hr)
        imgs_lr = normalize_m11(imgs_lr)
        return imgs_hr, tags_hr, imgs_lr, tags_lr

    def next(self):
        cur_store_idx = choice(range(len(self.data_store)))
        cur_store = self.data_store[cur_store_idx]
        length = cur_store['length']
        frame_idx = np.random.randint(0, length)
        res = self.gen_label((cur_store_idx, frame_idx))
        return res


if __name__ == '__main__':
    # g = C3DGenerator(c.agg_info_path, c.clip_shape, 1)
    # a = g.__next__()
    # g2 = SRGANGenerator(c.bmb_info_path, c.image_shape_hr, c.image_shape_lr, 1)
    # b = g2.__next__()
    # draw_pic(b[0][0], b[1][0][0])
    # draw_pic(b[2][0], b[3][0][0])
    # print(1)
    pass
