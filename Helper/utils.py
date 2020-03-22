import os
import cv2
from config import *
from datetime import datetime


def draw_pic(img, boxes=[], name=None):
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255,0, 255), (0, 255, 255)]
    for i in range(len(boxes)):
        box = boxes[i]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[i])
    now = datetime.now()
    if name is None:
        name = now.strftime("%m_%d_%H_%M_%S%f")
    file_name = os.path.join(test_dir, str(name) + '.jpg')
    cv2.imwrite(file_name, img)
    return


def save_weights(model, name):
    weight_path = os.path.join(weight_dir, name)
    model.save_weights(weight_path)
    return


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5