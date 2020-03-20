import config
from helper.pre_processing import *

if __name__ == '__main__':
    video2image(config.hr_video_path, config.hr_img_path, config.hr_video_info)
    video2image(config.video_path, config.img_path, config.video_info)
