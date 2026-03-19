import glob
from collections.abc import Mapping, Sequence
from datasets.trans import trans
# from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
# from mmcv.parallel import collate
import torch
import pandas as pd
import os
import numpy as np
import random
from PIL import Image
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
# PIPELINES = Registry('pipeline')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class CAPDataset(Dataset):
    def __init__(self,root_path, mode, num_segments, transform):
        self.root_path = root_path
        self.mode = mode  # 'training', 'testing', 'validation'
        # self.pipeline = Compose(pipeline)
        self.num_segments = num_segments
        self.transform = transform
        if self.mode in ["training", "validation"]:
            self._num_clips = 1
        elif self.mode in ["testing"]:
            self._num_clips = (
                    5*3
            )  # 5个样本每个样本3个

        self.labels, self.clips, self.text, self._spatial_temporal_idx, self.data_list = self.get_data_list()
        self.classes = self.get_classes_list()

    def get_classes_list(self):
        list_file = os.path.join(self.root_path, 'classes.txt')
        assert os.path.exists(list_file), "%s file does not exist!" % list_file
        self.classes = []
        with open(list_file, 'r', encoding='utf-8') as f:
            for ids, line in enumerate(f.readlines()):
                self.classes.append(line)
        return self.classes

    def get_data_list(self):
        list_file = os.path.join(self.root_path, self.mode + '.csv')
        assert os.path.exists(list_file), "%s file does not exist!" % list_file
        self.data_list = []
        self.clips = []
        self.labels = []
        self._spatial_temporal_idx = []
        self.text = []
        with open(list_file, 'r', encoding='utf-8') as f:
            for ids, line in enumerate(f.readlines()):
                sample = line.strip().split('+')  # 1/1_1_001537 49 56  ego-car hits a crossing pedestrian
                sample1 = sample[0].strip().split(' ')  # 1/1_1_001537 ，49， 56
                for idx in range(self._num_clips):
                    self.labels.append(int(sample1[0].split('/')[0]) - 1)
                    self.clips.append([int(sample1[1]), int(sample1[2])])
                    self.text.append(sample[1])
                    self._spatial_temporal_idx.append(idx)
                    self.data_list.append(sample1[0])  # 1/1_1_001537
        return self.labels, self.clips, self.text, self._spatial_temporal_idx, self.data_list

    def read_video(self, video_file, start, frame_count, num_segments):
        video_datas = []
        video_name = video_file[0].split("/")[-3]
        if start+7 <= frame_count:
            end = start + 7
        else:
            end = frame_count
        for fid in range(start - 1, end):
            # print(video_name," ", fid)
            video_data = video_file[fid]
            video_data = Image.open(video_data)
            # video_data.show()
            video_data = self.transform(video_data)
            # unloder = transforms.ToPILImage()
            # a = video_data
            # image = a.cpu().clone()
            # image = unloder(image)
            # image.show()
            #
            # aaaaa
            # video_data = np.asarray(video_data, np.float32)
            # video_data=transforms.ToTensor()
            video_datas.append(video_data)
        if len(video_datas) < num_segments:
            video_datas += list(video_datas[-1] for i in range(num_segments - (end - start)-1))
        # print("video_datas:", len(video_datas))
        assert len(video_datas) == num_segments, f"{video_name}'取样不是{num_segments}帧'"
        # video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
        frames = [np.asarray(frame) for frame in video_datas]
        frames = torch.as_tensor(np.stack(frames))
        return frames

    # def gather_info(self, index):
    #     accident_id = int(self.data_list[index].split('/')[0])
    #     video_id = int(self.data_list[index].split('/')[1])
    #     start, end = self.clips[index]
    #     data_info = [accident_id, video_id, start, end]
    #     # y = torch.tensor(self.labels[index], dtype=torch.float32)
    #
    #     data_info = torch.tensor(data_info)
    #     return data_info, y

    def __getitem__(self, index):
        start_, acc_id = self.clips[index]
        start_ = [i for i in range(start_, acc_id+1)]  # [42,49]
        start = int(random.sample(start_, 1)[0])
        video_path = os.path.join(self.root_path, self.mode, self.data_list[index], 'images')

        if self.mode == "testing":
            video_path = glob.glob(video_path + '/' + "*.jpg")
        else:
            video_path = glob.glob(video_path + '/' + "*.jpg")
        video_path = sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0])))
        frame_count = len(video_path)

        frames = self.read_video(video_path, start, frame_count, num_segments=self.num_segments)  # 8,720,1280,3
        # frames = self.pipeline(frames)
        label = self.labels[index]
        text = self.text[index]
        # data_info, y = self.gather_info(index)
        # print(data_info,":",start)
        # print(frames.shape)
        return frames, label, text

    def __len__(self):
        return len(self.data_list)







def build_dataloader(config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    train_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]
    transform = trans()
    train_data = CAPDataset(config.DATA.ROOT, "training", 8, transform=transform)
    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    # sampler_train = torch.utils.data.DistributedSampler(
    #     train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    # )
    train_loader = DataLoaderX(
        train_data,
        # sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
        # collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )

    val_pipeline = [
        dict(type='DecordInit'),
        dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, test_mode=True),
        dict(type='DecordDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[1] = dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=config.DATA.NUM_FRAMES, multiview=config.TEST.NUM_CLIP)

    val_data = CAPDataset(config.DATA.ROOT, "validation", 8, transform=transform)
    # indices = np.arange(dist.get_rank(), len(val_data), dist.get_world_size())
    # indices = np.arange(len(val_data))
    # sampler_val = SubsetRandomSampler(indices)
    val_loader = DataLoaderX(
        val_data,
        # sampler=sampler_val,
        batch_size=2,
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
        print(batch_data[2])
