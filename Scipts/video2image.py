import config
from helper.pre_processing import *

if __name__ == '__main__':
    video2image(config.bmb_video_dir, config.bmb_image_dir, config.bmb_info_path, extract_img=False)
    video2image(config.agg_video_dir, config.agg_image_dir, config.agg_info_path, extract_img=False)
