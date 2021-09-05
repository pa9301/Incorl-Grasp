import json
import os
from datetime import datetime

import cv2
import numpy as np


def random_crop(img, seg=None, dep=None):  # : 1280 720
    target_crop_w = 444  # 360 #472
    target_crop_h = 360  # 360 #472
    pad = int((target_crop_w - target_crop_h) / 2)

    temp_rgb_img = np.zeros((target_crop_w, target_crop_w, 4), dtype=np.uint8)
    temp_rgb_img.fill(255)
    temp_seg_img = np.zeros((target_crop_w, target_crop_w), dtype=np.uint8)
    temp_seg_img.fill(0)
    temp_dep_img = np.zeros((target_crop_w, target_crop_w), dtype=np.uint8)
    temp_dep_img.fill(0.9999)

    h, w, _ = img.shape
    c_x = w / 2
    c_y = h / 2

    crop_w = [int(c_x - (target_crop_w / 2)), int(c_x + (target_crop_w / 2))]
    crop_h = [int(c_y - (target_crop_h / 2)), int(c_y + (target_crop_h / 2))]

    cropped_w = [0, int(crop_w[1] - crop_w[0])]
    cropped_h = [0, int(crop_h[1] - crop_h[0])]

    temp_rgb_img[cropped_h[0] + pad:cropped_h[1] + pad, cropped_w[0]:cropped_w[1], :] \
        = img[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1], :]
    if seg is not None:
        temp_seg_img[cropped_h[0] + pad:cropped_h[1] + pad, cropped_w[0]:cropped_w[1]] \
            = seg[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]
    if dep is not None:
        temp_dep_img[cropped_h[0] + pad:cropped_h[1] + pad, cropped_w[0]:cropped_w[1]] \
            = dep[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]

    return temp_rgb_img, temp_seg_img, temp_dep_img


def img_normalize(img):
    img = img / 127.5 - 1.0
    return img


def img_invert_normalize(img, scale):
    img += 1
    img *= (scale / 2.0)
    img = img.astype('uint8')
    return img


def write_data(root_path, prev_state, after_state, action, prev_internal, after_internal, reward, terminal, idx):
    pid = os.getpid()
    now_time = datetime.now()
    str_now_time = now_time.strftime("%y%m%d_%H%M%S")
    file_name = '%d_%s_%d' % (pid, str_now_time, idx)

    data_dict = {'state before': prev_state,
                 'state after': after_state,
                 'action': action,
                 'internal state before': prev_internal,
                 'internal state after': after_internal,
                 'reward': reward,
                 'terminal': terminal}

    path = root_path + '/file/' + file_name + '.txt'

    with open(path, 'w') as f:
        f.write(json.dumps(data_dict))

    return path


def get_points(label_image_array, object_index):
    return np.argwhere(label_image_array == object_index)


def refine_segmented_image_by_connected_component(org_label_array, obj_id, num_noise_pixel):
    shape = (256, 256)
    binary_image_array = np.zeros(shape=shape, dtype=np.uint8)

    point_list = get_points(org_label_array, obj_id)
    binary_image_array.fill(0)

    binary_image_array[point_list[:, 0], point_list[:, 1]] = obj_id

    connectivity = 8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image_array, connectivity, cv2.CV_32S)

    if num_labels > 1:

        for m in range(0, num_labels):
            pixels_num = stats[m, cv2.CC_STAT_AREA]
            if pixels_num < num_noise_pixel:
                point_list2 = get_points(labels, m)
                binary_image_array[point_list2[:, 0], point_list2[:, 1]] = 0

    return binary_image_array
