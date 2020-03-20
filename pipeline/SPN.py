from configs.base_config import *
from helper.roi_helper import *
import math
import numpy as np
from pipeline.c3dgenrator import *
import random


class SPN(BaseConfig):
    def __init__(self, num_cls, img_shape):
        """

        @param base_net: Name of base net
        @param img_shape: Shape of input image (height, width)
        @param feature_shape: Feature Map shape, adjust according to base net (height, width)
        @param num_cls: classification class
        """
        super().__init__()
        self.downscale = 4
        self.anchor_box_scale = [16, 32, 48]
        self.anchor_box_ratio = [(1, 1), (1, 2), (2, 1), (math.sqrt(2), 1), (1, math.sqrt(2))]
        self.anchor_sets = [(scale, ratio) for scale in self.anchor_box_scale for ratio in self.anchor_box_ratio]

        self.num_anchors = len(self.anchor_sets)
        self.rpn_positive = 0.6
        self.rpn_negative = 0.3
        self.num_cls = num_cls
        self.threshold = 0.5
        self.img_shape = img_shape
        self.feature_shape = (self.img_shape[0] // self.downscale, self.img_shape[1] // self.downscale, 512)
        self.pooling_region = 5
        self.num_roi = 8
        self.num_rpn_valid = 128

    def cal_gt_tags(self, labels):
        """
        Calculate image ground truth value according to net config
        @param img: the input img
        @param labels: labels of the image
        @return: ground truth tag for rpn net
        """
        f_height, f_width = self.feature_shape[:2]  # feature map size
        num_bbox = len(labels)  # number of objects
        num_anchors = len(self.anchor_sets)

        box_valid = np.zeros((f_height, f_width, num_anchors))  # whether anchor box is valid(positive or negative)
        box_signal = np.zeros((f_height, f_width, num_anchors))  # anchor box label(0 for negative, 1 for positive)
        box_class = np.zeros((f_height, f_width, num_anchors))  # class for the box

        box_rpn_valid = np.zeros((f_height, f_width, 4 * num_anchors))  # whether rpn regression valid
        box_rpn_reg = np.zeros((f_height, f_width, 4 * num_anchors))  # rpn regression ground truth

        box_raw = np.zeros((f_height, f_width, 4 * num_anchors))  # raw anchor box without adjust

        num_anchors_for_bbox = np.zeros(num_bbox).astype(int)  # positive anchor for each bounding box

        # find the best anchor box for each bounding box, make sure there are at least 1 positive anchor box
        best_anchor_for_bbox = -1 * np.ones((num_bbox, 3)).astype(int)
        best_iou_for_bbox = np.zeros(num_bbox).astype(np.float32)
        best_x_for_bbox = np.zeros((num_bbox, 4)).astype(int)
        best_dx_for_bbox = np.zeros((num_bbox, 4)).astype(np.float32)

        # best iou of all boxes
        best_iou_all = 0
        for anchor_idx in range(len(self.anchor_sets)):

            anchor_scale, anchor_ratio = self.anchor_sets[anchor_idx]

            anchor_width = anchor_scale * anchor_ratio[0]
            anchor_height = anchor_scale * anchor_ratio[1]
            for ix in range(f_width):
                for jy in range(f_height):
                    # calculate anchor box coordinates
                    x1 = (ix + 0.5) * self.downscale - anchor_width / 2
                    y1 = (jy + 0.5) * self.downscale - anchor_height / 2

                    x2 = x1 + anchor_width
                    y2 = y1 + anchor_height
                    # if anchor box invalid, clip it to a valid shape
                    if x1 < 0 or y1 < 0 or x2 > self.img_shape[1] or y2 > self.img_shape[0]:
                        continue

                    # anchor box coordinates in raw image
                    anchor_box = (x1, y1, x2, y2)
                    box_raw[jy, ix, 4 * anchor_idx: 4 * anchor_idx + 4] = [x1, y1, x2, y2]

                    best_iou = 0.0  # current best IoU for this anchor box
                    best_class = -1  # current best class for this anchor box
                    for label_idx in range(len(labels)):
                        # for each object, calculate the iou between current anchor box and object
                        label = labels[label_idx]
                        gt_box = label
                        cur_iou = calculate_iou(gt_box, anchor_box)
                        if cur_iou > best_iou_all:
                            # record best iou of all anchors for debugging
                            best_iou_all = cur_iou
                        if cur_iou > best_iou:
                            best_iou = cur_iou
                            if cur_iou > self.rpn_positive:
                                # if higher than its best, box valid, label it positive
                                best_iou = cur_iou
                                best_class = label_idx

                                box_valid[jy, ix, anchor_idx] = 1
                                box_signal[jy, ix, anchor_idx] = 1
                                box_class[jy, ix, anchor_idx] = best_class

                                # calculate the rpn regression ground truth
                                box_rpn_valid[jy, ix, 4 * anchor_idx: 4 * anchor_idx + 4] = 1
                                box_rpn_reg[jy, ix, 4 * anchor_idx: 4 * anchor_idx + 4] = cal_dx(gt_box, anchor_box)
                                num_anchors_for_bbox[label_idx] += 1

                        if cur_iou > best_iou_for_bbox[label_idx]:
                            # id1f higher than the best iou of current object, record
                            best_iou_for_bbox[label_idx] = cur_iou
                            best_x_for_bbox[label_idx, :] = [x1, y1, x2, y2]

                            best_anchor_for_bbox[label_idx] = [jy, ix, anchor_idx]
                            best_dx_for_bbox[label_idx, :] = cal_dx(gt_box, anchor_box)

                    if best_iou < self.rpn_negative:
                        # if best iou of current anchor box less than threshold, box valid, label it negative
                        box_valid[jy, ix, anchor_idx] = 1
                        box_signal[jy, ix, anchor_idx] = 0
        #
        # for bbox_idx in range(num_bbox):
        #     # set anchor box each object
        #     cur_jy, cur_ix, cur_anchor_idx = best_anchor_for_bbox[bbox_idx]
        #     box_valid[cur_jy, cur_ix, cur_anchor_idx] = 1
        #     box_signal[cur_jy, cur_ix, cur_anchor_idx] = 1
        #
        #     box_rpn_valid[cur_jy, cur_ix, 4 * cur_anchor_idx: 4 * cur_anchor_idx + 4] = 1
        #     box_rpn_reg[cur_jy, cur_ix, 4 * cur_anchor_idx: 4 * cur_anchor_idx + 4] = best_dx_for_bbox[bbox_idx]
        #
        #     box_class[cur_jy, cur_ix, cur_anchor_idx] = labels[bbox_idx]['category']

        pos_loc = np.where(np.logical_and(box_valid[:, :, :] == 1, box_signal[:, :, :] == 1))
        neg_loc = np.where(np.logical_and(box_valid[:, :, :] == 1, box_signal[:, :, :] == 0))

        num_pos = len(pos_loc[0])
        num_neg = len(neg_loc[0])

        if num_pos > self.num_rpn_valid / 2:
            val_loc = random.sample(range(len(neg_loc[0])), self.num_rpn_valid)
            box_valid[pos_loc[0][val_loc], pos_loc[1][val_loc], pos_loc[2][val_loc]] = 0

        if num_neg > num_pos:
            val_loc = random.sample(range(len(neg_loc[0])), num_neg - num_pos)
            box_valid[neg_loc[0][val_loc], neg_loc[1][val_loc], neg_loc[2][val_loc]] = 0

        rpn_cls = np.concatenate([box_valid, box_signal], axis=2)
        rpn_reg = np.concatenate([box_rpn_valid, box_rpn_reg], axis=2)

        # rpn_cls = np.array(box_signal)
        # rpn_reg = np.array(box_rpn_reg)

        # print(num_pos)
        rpn_cls_valid = sum(sum(sum(box_rpn_valid)))
        return rpn_cls, rpn_reg, box_class, box_raw, rpn_cls_valid
