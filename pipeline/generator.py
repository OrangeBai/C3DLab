import os
import config
import json
import math
import cv2
import numpy as np
from random import *


class Generator:
    def __init__(self):
        self.data_store = None
        self.clip_length = 5
        self.action_counter = []
        self.action_indexer = {}
        self.__load_info__()
        self.unzip_label()

    def __load_info__(self):
        with open(config.video_info, 'r') as file:
            self.data_store = json.load(file)

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
        video_idx, pic_idx = clip_idx
        cur_clip = []
        cur_tag = {'action': [], 'pos': []}
        cur_video_info = self.data_store[video_idx]
        video_length = len(cur_video_info['label'][0][0])
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
        cur_clip = np.expand_dims(cur_clip, 0)
        cur_label = [np.array(cur_tag['pos']).transpose([1, 0, 2]), np.array(cur_tag['action'])]

        return cur_clip, cur_label

    def next(self):
        return self.__next__()

    def __next__(self):
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


if __name__ == '__main__':
    g = Generator()
    for i in range(10000):
        next(g)
        print(i)
    print(1)
