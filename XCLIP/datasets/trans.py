from torchvision import transforms
import random
import numpy as np
import torch


# scale_resize = int(256 / 224 *224)
# scales = (1, 0.875, 0.75, 0.66)
# input_size = (224, 224)
# max_wh_scale_gap = 1
# base_size = min(scale_resize, scale_resize)
# crop_sizes = [int(base_size * s) for s in scales]
#
# candidate_sizes = []
# for i, h in enumerate(crop_sizes):
#     for j, w in enumerate(crop_sizes):
#         if abs(i - j) <= max_wh_scale_gap:
#             candidate_sizes.append([w, h])
# crop_size = random.choice(candidate_sizes)
# for i in range(2):
#     if abs(crop_size[i] - input_size[i]) < 3:
#         crop_size[i] = input_size[i]
#
# crop_w, crop_h = crop_size
# w_step = (scale_resize - crop_w) // 4
# h_step = (scale_resize - crop_h) // 4
# candidate_offsets = [
#     (0, 0),  # upper left
#     (4 * w_step, 0),  # upper right
#     (0, 4 * h_step),  # lower left
#     (4 * w_step, 4 * h_step),  # lower right
#     (2 * w_step, 2 * h_step),  # center
# ]
# x_offset, y_offset = random.choice(candidate_offsets)
#


def trans():
    scale_resize = int(256 / 224 * 224)
    scales = (1, 0.875, 0.75, 0.66)
    input_size = (224, 224)
    max_wh_scale_gap = 1
    base_size = min(scale_resize, scale_resize)
    crop_sizes = [int(base_size * s) for s in scales]

    candidate_sizes = []
    for i, h in enumerate(crop_sizes):
        for j, w in enumerate(crop_sizes):
            if abs(i - j) <= max_wh_scale_gap:
                candidate_sizes.append([w, h])
    crop_size = random.choice(candidate_sizes)
    for i in range(2):
        if abs(crop_size[i] - input_size[i]) < 3:
            crop_size[i] = input_size[i]

    crop_w, crop_h = crop_size
    w_step = (scale_resize - crop_w) // 4
    h_step = (scale_resize - crop_h) // 4
    candidate_offsets = [
        (0, 0),  # upper left
        (4 * w_step, 0),  # upper right
        (0, 4 * h_step),  # lower left
        (4 * w_step, 4 * h_step),  # lower right
        (2 * w_step, 2 * h_step),  # center
    ]
    x_offset, y_offset = random.choice(candidate_offsets)
    p = 0.8
    v = random.random()
    if v < p:
        transform = transforms.Compose([
            # transforms.Resize((scale_resize, scale_resize)),
            # transforms.RandomCrop(x_offset, y_offset),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            # transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
            transforms.ToTensor(),
            transforms.Normalize([0.403932, 0.42002773, 0.42598072], [0.25544345, 0.2625126, 0.27170303])
        ])
    else:
        transform = transforms.Compose([
            # transforms.Resize((scale_resize, scale_resize)),
            # transforms.RandomCrop(x_offset, y_offset),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.Normalize([103.00266, 107.10707115, 108.6250836],[65.13807975, 66.940713, 69.28427265]),
            transforms.ToTensor(),
            transforms.Normalize([0.403932, 0.42002773, 0.42598072],[0.25544345, 0.2625126, 0.27170303])
        ])

    return transform

# from PIL import Image
# a = Image.open(r"F:\CAP-DATA\testing\1\001537\images\000001.jpg")
# b = trans()
# c = b(a)
# print(c.shape)
