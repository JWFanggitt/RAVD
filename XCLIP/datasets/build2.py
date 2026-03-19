import glob
import math

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate
# from mmcv.parallel import collate
import torch
import pandas as pd
import os
import numpy as np
import random
import json
from PIL import Image
from prefetch_generator import BackgroundGenerator
from mydataset import CAPDATA

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
# PIPELINES = Registry('pipeline')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class CAPDataset(Dataset):
    def __init__(self, root: str, num_segments: int, num_sample: int, mode: str, transform):
        self.root = root
        self.mode = mode
        self.num_segments = num_segments
        self.num_sample = num_sample
        self.name_map = json.load(open('/home/ubuntu/CAP-DATA/classes.json'))
        self.transform = transform
        self.data_list, self.texts, self.clips, self.labels = self.get_data_list()
        self.classes = self.get_classes_list()

    def get_data_list(self):
        list_file = os.path.join(self.root, self.mode + f'-{self.num_segments}.csv')
        self.labels = []
        self.data_list = []
        self.texts = []
        self.clips = []
        with open(list_file, 'r', encoding='utf-8') as f:
            for ids, line in enumerate(f.readlines()):
                sample = line.strip().split('+')  # 1/1_1_001537 49 56  ego-car hits a crossing pedestrian
                sample1 = sample[0].strip().split(' ')  # 1/1_1_001537 ，49， 56
                text = sample[1].replace('\xa0', ' ')  # ego-car hits a crossing pedestrian
                text.strip()
                self.labels.append(int(sample1[0].split('/')[0]) - 1)
                self.data_list.append(sample1[0])  # 1/1_1_001537
                self.clips.append([int(sample1[1]), int(sample1[2])])
                self.texts.append(text)
        # self.video_path = [self.root + '/' + self.mode + '/' + vid + '.npy' for vid in self.data_list]
        #  root/mode/1/1_1_001537.npy
        return self.data_list, self.texts, self.clips, self.labels

    def __len__(self):
        return len(self.data_list)

    def get_classes_list(self):
        list_file = os.path.join(self.root, 'classes.txt')
        assert os.path.exists(list_file), "%s file does not exist!" % list_file
        self.classes = []
        with open(list_file, 'r', encoding='utf-8') as f:
            for ids, line in enumerate(f.readlines()):
                self.classes.append(line)
        return self.classes

    def get_frames(self, video_file, start, frame_count, num_segments):
        video_name = video_file[0].split("/")[-3]
        # print(video_name, frame_count)
        tmp_repos = []
        for index, value in enumerate(start):
            if start[index] + num_segments-1 <= frame_count:
                end = start[index] + num_segments-1
            else:
                end = frame_count
            video_clip = []
            for fid in range(value - 1, end):
                # print(video_name, fid)
                video_data = video_file[fid]
                video_data = Image.open(video_data)
                video_data = self.transform(video_data)
                video_clip.append(video_data)
            if len(video_clip) < num_segments:
                video_clip += list(video_clip[-1] for i in range(num_segments - int(len(video_clip))))
            assert len(video_clip) == num_segments, f"{video_name}'取样不是{num_segments}帧'"
            frames = [np.asarray(frame) for frame in video_clip]
            frames = torch.as_tensor(np.stack(frames))  # 8,3,720,1280
            tmp_repos.append(frames)
        if self.mode == 'training':
            tensor_clip = tmp_repos[0]  # 8,3,720,1280
            # tensor_clip = tensor_clip.permute(0,1,4,2,3)
        else:
            tensor_clip = torch.stack(tmp_repos, dim=0)
            # tensor_clip = tensor_clip.permute(0,3,1,2)
        return tensor_clip

    def __getitem__(self, idx):
        action_id = self.data_list[idx].split("/")[0]
        action_name = self.classes[int(action_id)-1]
        video_path = os.path.join(self.root, self.mode, self.data_list[idx], 'images')
        video_file = glob.glob(video_path + '/' + "*.[jp][pn]g")
        video_file = sorted(video_file, key=lambda x: int((os.path.basename(x).split('.')[0])))
        frame_count = len(video_file)
        start, acc_id = self.clips[idx]
        if acc_id > frame_count:
            acc_id = frame_count
        start_new = [i for i in range(start, acc_id + 1)]
        if len(start_new) >= self.num_sample:
            start = random.sample(start_new, self.num_sample)
        else:
            start = np.random.choice(start_new, self.num_sample)
        start.sort()
        label = self.labels[idx]
        text = self.texts[idx]
        frames = self.get_frames(video_file, start, frame_count, num_segments=self.num_segments)
        # print(frames.shape)
        return frames, label, text


def random_short_side_scale_jitter(images, size, inverse_uniform_sampling=False):
    if inverse_uniform_sampling:
        size = int(round(1.0 / np.random.uniform(1.0 / size[1], 1.0 / size[0], )))
    else:
        size = int(round(np.random.uniform(size[0], size[1])))
    images = torch.unsqueeze(images, 0)
    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
            height <= width and height == size
    ):
        return images
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    random_short_side_scale_jitter_image = torch.nn.functional.interpolate(images, size=(new_height, new_width),
                                                                           mode="bilinear", align_corners=False)
    return random_short_side_scale_jitter_image


def random_crop(images, size):
    if images.shape[2] == size and images.shape[3] == size:
        return images, None
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[:, :, y_offset: y_offset + size, x_offset: x_offset + size]
    cropped = torch.squeeze(cropped)
    return cropped


def build_dataloader(config):
    # scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)
    transform_train = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                                        saturation=0.2, hue=0.1)], p=0.8),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         # transforms.Normalize([0.451729, 0.7094968, 0.47192925], [0.3002181, 0.30520406, 0.31152505]),
         # transforms.Lambda(lambda x: x + config.AUG.SIGMA * torch.randn_like(x)),
         transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 1.0))
         ]
    )
    transform_val = transforms.Compose(
        [transforms.Resize([224, 224]),
         # transforms.RandomCrop([224,224]),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         # transforms.Normalize([0.451729, 0.7094968, 0.47192925], [0.3002181, 0.30520406, 0.31152505]),
         ]
    )
    # transform_val = transforms.Compose(
    #     [
    #         # transforms.Resize([224, 224]),
    #         transforms.ToTensor(),
    #         transforms.Lambda(lambda x: random_short_side_scale_jitter(x, [256,224])),
    #         transforms.Lambda(lambda x: random_crop(x, 224)),
    #         # transforms.Normalize([0.451729, 0.7094968, 0.47192925], [0.3002181, 0.30520406, 0.31152505]),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]
    # )

    train_data = CAPDataset(config.DATA.ROOT, config.DATA.NUM_FRAMES, 1,
                            "training", transform=transform_train)
    train_loader = DataLoaderX(
        train_data,
        # sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        # batch_size=1,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=False,
        # collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )

    val_data = CAPDataset(config.DATA.ROOT, config.DATA.NUM_FRAMES, 1,
                          "testing", transform=transform_val)
    # val_data = CAPDATA(config.DATA.ROOT, "validation", 8, 1, transform=transform_val)

    val_loader = DataLoaderX(
        val_data,
        # sampler=sampler_val,
        batch_size=4,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        # collate_fn=partial(mmcv_collate, samples_per_gpu=2),
    )

    return train_data, val_data, train_loader, val_loader


if __name__ == "__main__":
    import os
    import torch
    import torch.distributed as dist
    import argparse
    from utils.config import get_config
    import numpy as np
    import random
    from utils.config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default=r'D:\X-CLIP\configs\k400\16_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="./results")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)

    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    args = parser.parse_args()

    config = get_config(args)
    train_data, val_data, train_loader, val_loader = build_dataloader(config)
    for idx, batch_data in enumerate(train_loader):
        print(batch_data[1])
        # print(batch_data[2])
