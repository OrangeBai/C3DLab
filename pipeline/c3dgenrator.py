import os
import config
import json
import math
import cv2
import numpy as np
from random import randint, choice


class DataGenerator:
    def __init__(self, info_path, batch_size=1):
        self.data_store = None
        self.__load_info__(info_path)
        self.batch_size = batch_size

    def __load_info__(self, info_path):
        with open(info_path, 'r') as file:
            self.data_store = json.load(file)

    def __next__(self):
        pass

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class C3DGenerator(DataGenerator):
    def __init__(self, info_path, batch_size=1):
        super().__init__(info_path, batch_size)
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
        video_idx, pic_idx = clip_idx
        cur_clip = []
        cur_tag = {'action': [], 'pos': []}
        cur_video_info = self.data_store[video_idx]
        video_length = len(cur_video_info['label'][0][0])
        st_idx = pic_idx - math.ceil((self.clip_length - 1) / 2)
        ed_idx = pic_idx + math.ceil((self.clip_length - 1) / 2)
        if ed_idx >= video_length or st_idx < 0:
            return None
        for frame_idx in range(st_idx, ed_idx + 1):
            cur_img_path = os.path.join(cur_video_info['img_dir'], str(frame_idx).zfill(6) + '.jpg')
            cur_img = cv2.imread(cur_img_path)
            cur_clip.append(cur_img)

        video_label = cur_video_info['label']
        for fly_idx in range(2):
            cur_tag['pos'].append(video_label[1][fly_idx][st_idx: ed_idx + 1])
            cur_tag['action'].append(video_label[0][fly_idx][pic_idx])

        cur_clip = np.array(cur_clip)
        cur_clip = np.expand_dims(cur_clip, 0)
        cur_label = [np.array(cur_tag['pos']).transpose([1, 0, 2]), np.array(cur_tag['action'])]

        return cur_clip, cur_label

    def __next__(self):
        return self.next()

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
    def __init__(self, info_path, batch_size=1):
        super().__init__(info_path, batch_size)

    def gen_label(self, clip_idx):
        video_idx, pic_idx = clip_idx
        cur_tag = {'action': [], 'pos': []}
        cur_video_info = self.data_store[video_idx]
        cur_img_path = os.path.join(cur_video_info['img_dir'], str(pic_idx).zfill(6) + '.jpg')
        cur_img = cv2.imread(cur_img_path)
        video_label = cur_video_info['label']
        for i in range(2):
            cur_tag['pos'].append(video_label[1][i][pic_idx])
            cur_tag['action'].append(video_label[0][i][pic_idx])

        return np.array(cur_img), [np.array(cur_tag['pos']), np.array(cur_tag['action'])]

    def __next__(self):
        imgs = []
        pos = []
        action = []
        for i in range(self.batch_size):
            img, tag = self.next()
            imgs.append(img)
            pos.append(tag[0])
            action.append(tag[1])
        imgs = np.array(imgs).astype('float32')
        pos = np.array(pos).astype('float32')
        action = np.array(action).astype('float32')
        if self.batch_size == 1:
            imgs = np.expand_dims(imgs, axis=0)
            pos = np.expand_dims(pos, axis=0)
            action = np.expand_dims(action, axis=0)
        return imgs, [pos, action]

    def next(self):
        cur_store_idx = choice(range(len(self.data_store)))
        cur_store = self.data_store[cur_store_idx]
        length = cur_store['length']
        frame_idx = np.random.randint(0, length)
        res = self.gen_label((cur_store_idx, frame_idx))
        return res


if __name__ == '__main__':
    # g = C3DGenerator(config.video_info, batch_size=1)
    # for i in range(10):
    #     b = next(g)
    #     print(i)
    # print(1)
    g2 = SRGANGenerator(config.hr_video_info, 4)
    g = iter(g2)
    a = next(g)
    print(1)
    pass
