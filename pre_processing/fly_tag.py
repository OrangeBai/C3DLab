from scipy.io import loadmat
import pickle
import os
import numpy as np


def generate_pickle(video_info):
    """
    Generate tags from mat files which are original tags of the data set
    :param video_info: Video info tuple (Video Name, Video Path, Output_Path)
    :return: The tag file: output_path/Info.p, which is a pickle file, should be read by Info_reader()
    """
    output_path = video_info[2]

    video_name = os.path.splitext(video_info[0])[0]
    action_name = video_name + '_actions.mat'
    feat_name = video_name + '_feat.mat'
    track_name = video_name + '_track.mat'

    action_path = os.path.join(video_info[1], action_name)
    feat_path = os.path.join(video_info[1], feat_name)
    track_path = os.path.join(video_info[1], track_name)

    action_mat = loadmat(action_path)
    beh = action_mat['behs'].tolist()[0]
    bouts = action_mat['bouts'].tolist()
    action_list = {'behs': beh, 'bouts': bouts}

    track = loadmat(track_path)
    trk = track['trk']
    flag_frames = trk[0, 0]['flag_frames'].tolist()
    names = trk[0, 0]['names'][0].tolist()
    data = trk[0, 0]['data'].tolist()
    track_list = {'flag_frames': flag_frames, 'names': names, 'data': data}

    feat = loadmat(feat_path)
    feat_names = feat['feat'][0, 0]['names'].tolist()
    feat_data = feat['feat'][0, 0]['data'].tolist()
    feat_list = {'feat_names': feat_names, 'data_names': feat_data}

    pickle_stream = {'action': action_list, 'track': track_list, 'feat': feat_list}

    pickle_path = os.path.join(output_path, 'tags.p')
    with open(pickle_path, 'wb') as writer:
        pickle.dump(pickle_stream, writer)

    return pickle_path


def get_action_and_tags(base_path):
    """
    Get action names and tags of all videos
    :param base_path: Base path of all videos
    :return: action_dir:    Dictionary with {action_name: data}
                             data is list of tuples: [(Video_frame_path, fly_number, frame_number)]
              action: list of action_names
    """
    dir_lists = os.listdir(base_path)
    action_dir = {}
    bouts = []
    actions = None
    for d in dir_lists:
        cur_path = os.path.join(base_path, d)
        # Check if the path is a directory
        if os.path.isdir(cur_path):
            pkl_file_path = os.path.join(cur_path, 'tags.p')
            try:
                # Load the pickle file
                with open(pkl_file_path, 'rb') as reader:
                    info = pickle.load(reader)
                # If actions is None, load the action names
                if actions is None:
                    temp_actions = []
                    for action in info['action']['behs']:
                        temp_actions.append(action[0])
                    actions = temp_actions
                    bouts = [[] for _ in actions]
                cur_bouts = action_bouts_in_one_video(cur_path)
                for i in range(len(cur_bouts)):
                    bouts[i].extend(cur_bouts[i])
            except FileNotFoundError as e:
                print(e)
    for i in range(len(actions)):
        action_dir[actions[i]] = bouts[i]
    return action_dir, actions


def action_bouts_in_one_video(pickle_base_path):
    """
    Get all action bouts of a video
    :param pickle_base_path: path of a frame directory, which should contain the information pickle file
    :return: frames of occurring action,
    """
    file_path = os.path.join(pickle_base_path, 'tags.p')
    assert os.path.exists(pickle_base_path)
    # Open the pickle file of a video
    with open(file_path, 'rb') as reader:
        info = pickle.load(reader)
    action_frames = [[] for _ in range(len(info['action']['bouts'][0]))]
    bouts = info['action']['bouts']
    for fly_num in range(len(bouts)):
        for action in range(len(bouts[fly_num])):
            for bout in bouts[fly_num][action]:
                for frame in range(bout[0], bout[1] + 1):
                    act = (pickle_base_path, fly_num, frame)
                    action_frames[action].append(act)
    return action_frames


def get_pos(base_path, only_pos=True):
    """
    Get the position information of a given file
    :param base_path: path of a frame directory, which should contain the information pickle file
    :param only_pos: If true, only extract position tags
    :return: Frame_Path: positions dictionary:
              FileDirectory: position info( 2 * 2 * 5400)
    """
    dir_lists = os.listdir(base_path)
    pos_dir = {}
    for d in dir_lists:
        cur_path = os.path.join(base_path, d)
        # Check if the path is a directory
        if os.path.isdir(cur_path):
            pkl_file_path = os.path.join(cur_path, 'tags.p')
            if not os.path.exists(pkl_file_path):
                print('No pickle file in {0}'.format(cur_path))
                continue
            try:
                # Load the pickle file
                cur_pos = pos_tags_in_one_video(cur_path, only_pos)
                pos_dir[cur_path] = cur_pos
            except FileNotFoundError as e:
                print(e)
    return pos_dir


def pos_tags_in_one_video(pickle_base_path, only_pos=True):
    """
    Get position information of a video from pickle file
    :param pickle_base_path: The base path of the pickle file
    :return: Position information of flies
    """
    file_path = os.path.join(pickle_base_path, 'tags.p')
    assert os.path.exists(pickle_base_path)
    with open(file_path, 'rb') as reader:
        info = pickle.load(reader)
    pos = np.array(info['track']['data'])
    if only_pos:
        pos = pos[:, :, :2]
    return pos


def check_action(action_frames, frame_tuple):
    """
    Check whether a fly in a frame is performing an action or not.
    :param action_frames: The list that recording all actions
    :param frame_tuple: Frame tuple: (Video_base_path, fly_num, frame_num)
    :return: if -1, fly is normal, else return fly_tag
    """
    for action_name in action_frames[1]:
        if frame_tuple in action_frames[0][action_name]:
            return action_frames[1].index(action_name)
    return -1


def save_tags_and_info(base_path, pos, actions):
    pos_pickle = os.path.join(base_path, 'pos.p')
    action_pickle = os.path.join(base_path, 'action.p')
    with open(pos_pickle, 'wb') as writer:
        pickle.dump(pos, writer)
    with open(action_pickle, 'wb') as writer:
        pickle.dump(actions, writer)


def load_tags_and_info(base_path):
    """
    Read Pickle file to get actions and positions information
    :param base_path: path of pickle file
    :return: Position information and action file
    """
    pos_pickle = os.path.join(base_path, 'pos.p')
    action_pickle = os.path.join(base_path, 'action.p')
    with open(pos_pickle, 'rb') as reader:
        pos = pickle.load(reader)
    with open(action_pickle, 'rb') as reader:
        actions = pickle.load(reader)
    return pos, actions
