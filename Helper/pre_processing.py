import os
import cv2
from scipy.io import loadmat
import math
import numpy as np
import json


def video2image(input_path, output_path, info_path, extract_img=True):
    """
    Extract images from video, load json files
    :param input_path: Video base path
    :param output_path: Image output path
    :param info_path: Info path
    :param extract_img: Boolean, if true, extract all images
    :return: None
    """
    video_files = []
    info = {
        'action': '_actions.mat',
        'feat': '_feat.mat',
        'track': '_track.mat'
    }
    for r, d, f in os.walk(input_path):
        for file in f:
            file_ext = os.path.splitext(file)[-1]
            if file_ext in ['.mp4', '.avi']:
                file_name = os.path.splitext(file)[0]
                if file_name + info['action'] in f and file_name + info['feat'] in f and file_name + info['track'] in f:
                    video_path = os.path.join(r, file)
                    cap = cv2.VideoCapture(video_path)
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cur_video = {
                        'name': file_name,  # file_name
                        'path': video_path,  # file_path
                        'dir': r,  # directory
                        'action': os.path.join(r, file_name + info['action']),  # action mat path
                        'feat': os.path.join(r, file_name + info['feat']),  # feature mat path
                        'track': os.path.join(r, file_name + info['track']),  # track mat path
                        'length': length
                    }
                    img_dir = os.path.join(output_path, file_name)
                    # while os.path.exists(img_dir):
                    #     img_dir = img_dir + '_'
                    cur_video['img_dir'] = img_dir
                    video_files.append(cur_video)

    for video_info in video_files:
        data = generate_pickle(video_info)
        video_info['label'] = data[:2]
        if extract_img:
            extract_video(video_info['path'], video_info['img_dir'], 6)

    with open(info_path, 'w') as f:
        json.dump(video_files, f)

    return


def extract_video(video_path, out_dir, name_length, ext='.jpg'):
    """
    retrieve all frames of an video
    :param video_path: path of video
    :param out_dir: directory of output images
    :param name_length: name length of video
    :param ext: extension of image
    :return: None
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    video_ext = os.path.splitext(video_path)[-1]
    assert video_ext in ['.mp4', '.avi']
    cap = cv2.VideoCapture(video_path)
    counter = 0
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            file_name = str(counter).zfill(name_length) + ext
            file_path = os.path.join(out_dir, file_name)
            cv2.imwrite(file_path, frame)
            counter = counter + 1
            l1 = counter * 50 // length
            print('Extracting {0}, Progress Bar: {1:><{len1}}{2:=<{len2}}. '
                  '{3} of {4}'.format(video_path, '>', '=', counter, length, len1=l1, len2=50 - l1, ))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    return


def generate_pickle(video_info):
    """
    Generate tags from mat files which are original tags of the data set
    :param video_info: Video info tuple (Video Name, Video Path, Output_Path)
    :return: The tag file: output_path/Info.p, which is a pickle file, should be read by Info_reader()
    """
    action_mat = loadmat(video_info['action'])
    beh = action_mat['behs'].tolist()[0]
    bouts = action_mat['bouts'].tolist()
    action_list = {'behs': beh, 'bouts': bouts}

    track_mat = loadmat(video_info['track'])
    trk = track_mat['trk']
    flag_frames = trk[0, 0]['flag_frames'].tolist()
    names = trk[0, 0]['names'][0].tolist()
    data = trk[0, 0]['data'].tolist()
    track_list = {'flag_frames': flag_frames, 'names': names, 'data': data}

    feat_mat = loadmat(video_info['feat'])
    feat_names = feat_mat['feat'][0, 0]['names'].tolist()
    feat_data = feat_mat['feat'][0, 0]['data'].tolist()
    feat_list = {'feat_names': feat_names, 'data_names': feat_data}

    pickle_stream = {'action': action_list, 'track': track_list, 'feat': feat_list}

    video_length = len(data[0])
    beh = np.zeros((2, video_length))
    for i in range(len(bouts)):
        cur_fly = bouts[i]
        for j in range(len(cur_fly)):
            cur_bouts = cur_fly[j]
            for bout in cur_bouts:
                for frame in range(bout[0], bout[1] + 1):
                    beh[i][frame] = j + 1
    beh = beh.tolist()
    loc = np.zeros((2, video_length, 4))
    for i in range(len(data)):
        cur_fly = data[i]
        for frame in range(len(cur_fly)):
            cur_track_data = cur_fly[frame]
            loc[i, frame, :] = cal_bbox(cur_track_data)
    loc = loc.tolist()
    return beh, loc, pickle_stream


def cal_bbox(data):
    """
    transfer track data into bounding boxes
    :param data: x_center, y_center, ori, max_axis_length, min_axis_length
    :return: bounding box
    """
    xc, yc, ori, l, d = data[:5]
    w1, w2 = data[-2:]
    if not math.isnan(w1) and not math.isnan(w2):
        l = l + (w1 + w2) / 3
        d = d + (w1 + w2) / 3
    ori = ori % math.pi
    if 0 <= ori < math.pi / 2:
        h = l / 2 * math.sin(ori) + d / 2 * math.cos(ori)
        w = l / 2 * math.cos(ori) + d / 2 * math.sin(ori)
        bbox = [xc - w, yc - h, xc + w, yc + h]
    else:
        h = l / 2 * math.sin(math.pi - ori) + d / 2 * math.cos(math.pi - ori)
        w = l / 2 * math.cos(math.pi - ori) + d / 2 * math.sin(math.pi - ori)
        bbox = [xc - w, yc - h, xc + w, yc + h]
    for i in range(4):
        cor = bbox[i]
        if math.isnan(cor):
            cor = -1
        else:
            cor = round(cor)
        bbox[i] = cor
    return bbox
