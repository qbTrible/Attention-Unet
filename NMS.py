# -*- coding: utf-8 -*-
# @Time: 2020-02-11 15:09
# Author: Trible

import numpy as np
import cv2
import os

def distance(a_dot, b_dots):
    dis = np.sqrt((a_dot[0] - b_dots[:, 0])**2 + (a_dot[1] - b_dots[:, 1])**2)
    return dis

def index_nms(matrix, bri_thresh=200, dis_thresh=10):

    if matrix.shape[0] == 0:
        return np.array([])

    index = np.argwhere(matrix > bri_thresh)
    value = matrix[matrix > bri_thresh]
    index = index[value.argsort()]
    index_list = []

    while index.shape[0] > 1:
        a_dot = index[0]
        b_dots = index[1:]
        index_list.append(a_dot)
        dis = distance(a_dot, b_dots)
        index = b_dots[np.where(dis > dis_thresh)]
    if index.shape[0] > 0:
        index_list.append(index[0])

    return np.stack(index_list)

# 聚类螺钉
def bolt_nms(index, dis_thresh=100):
    bolt_list = []
    while index.shape[0] >= 3:
        a_dot = index[0]
        b_dots = index[0:]
        dis = distance(a_dot, b_dots)
        _index = b_dots[np.where(dis < dis_thresh)]
        if _index.shape[0] >= 3:
            bolt_list.append(_index)
        index = b_dots[np.where(dis > dis_thresh)]
    return bolt_list

# label_path = r"D:\Data\ann"
# pic_path = r"D:\Data\ori_pic"
# label = os.listdir(label_path)
# pic = os.listdir(pic_path)
#
# n = 0
# for file in label:
#     print(file)
#     y = cv2.imread(os.path.join(label_path, file), cv2.IMREAD_GRAYSCALE)
#     x = cv2.imread(os.path.join(pic_path, file.split(".")[0]+".jpg"), cv2.IMREAD_GRAYSCALE)
#     x = cv2.resize(x, (int(x.shape[0] * 1024 / 1808), int(x.shape[0] * 1024 / 1808)))
#     index = index_nms(y)
#     bolts = bolt_nms(index)
#     for bolt in bolts:
#         max_x = np.max(bolt[:, 1]) + 15
#         max_y = np.max(bolt[:, 0]) + 15
#         min_x = np.max((np.min(bolt[:, 1]) - 15, 0))
#         min_y = np.max((np.min(bolt[:, 0]) - 15, 0))
#         # print(min_y, max_y, min_x, max_x)
#         crop_img = x[min_y:max_y, min_x:max_x]
#         # print(crop_img.shape)
#         crop_img = cv2.resize(crop_img, (64, 64))
#         crop_label = y[min_y:max_y, min_x:max_x]
#         crop_label = cv2.resize(crop_label, (64, 64))
#         add_img = cv2.addWeighted(crop_img, 0.3, crop_label, 0.7, 0)
#         add_img = cv2.resize(add_img, (64, 64))
#         # print(crop_img.shape)
#         cv2.imwrite(r"D:\Data\bolt_pic\%d.jpg" % n, crop_img)
#         cv2.imwrite(r"D:\Data\bolt_label\%d.jpg" % n, crop_label)
#         cv2.imwrite(r"D:\Data\add_pic\%d.jpg" % n, add_img)
#         print("第%d张图片"%n)
#         n += 1

label_path = r"D:\Data\ann"
label = os.listdir(label_path)
with open(r"D:\Data\label.txt", "a+") as f:
    for file in label:
        print(file)
        f.write("ori_pic/%s.jpg" % file.split(".")[0])
        y = cv2.imread(os.path.join(label_path, file), cv2.IMREAD_GRAYSCALE)
        index = index_nms(y)
        bolts = bolt_nms(index)
        n = 0
        for bolt in bolts:
            max_x = np.max(bolt[:, 1]) + 15
            max_y = np.max(bolt[:, 0]) + 15
            min_x = np.min(bolt[:, 1]) - 15
            min_y = np.min(bolt[:, 0]) - 15

            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2
            width = max_x - min_x
            height = max_y - min_y

            f.write(" " + str(n) + " " + str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height))
            n += 1
        f.write("\n")
