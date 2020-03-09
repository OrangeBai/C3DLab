import config
import json
from pre_processing.util import *

video_files = []
info = {
    'action': '_actions.mat',
    'feat': '_feat.mat',
    'track': '_track.mat'
}
for r, d, f in os.walk(config.video_path):
    for file in f:
        if '.mp4' in file:
            file_name = os.path.splitext(file)[0]
            if file_name + info['action'] in f and file_name + info['feat'] in f and file_name + info['track'] in f:
                video_path = os.path.join(r, file)
                cap = cv2.VideoCapture(video_path)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cur_video = {
                    'name': file_name,  # file_name
                    'path': video_path,  # file_path
                    'dir': r,                                               # directory
                    'action': os.path.join(r, file_name + info['action']),  # action mat path
                    'feat': os.path.join(r, file_name + info['feat']),      # feature mat path
                    'track': os.path.join(r, file_name + info['track']),    # track mat path
                    'length':length
                }
                img_dir = os.path.join(config.img_path, file_name)
                # while os.path.exists(img_dir):
                #     img_dir = img_dir + '_'
                cur_video['img_dir'] = img_dir
                video_files.append(cur_video)


for video_info in video_files:
    data = generate_pickle(video_info)
    video_info['label'] = data[:2]
    # extract_video(video_info['path'], video_info['img_dir'], 6)

with open(config.video_info, 'w') as f:
    json.dump(video_files, f)
print(1)
