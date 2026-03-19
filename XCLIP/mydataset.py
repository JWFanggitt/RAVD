import glob
import os
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CAPDATA(Dataset):
    def __init__(self, root_path, phase, num_segments, interval=1, transform=None):
        self.root_path = root_path
        self.phase = phase  # 'training', 'testing', 'validation'
        self.interval = interval
        self.transforms = transform
        self.num_segments = num_segments
        self.classes = self.get_classes_list()
        self.data_list, self.labels, self.clips, self.texts = self.get_data_list()

    def get_data_list(self):
        global sample1
        list_file = os.path.join(self.root_path, self.phase + '.txt')
        assert os.path.exists(list_file), "%s file does not exist!" % list_file
        data_list, labels, clips, texts = [], [], [], []
        with open(list_file, 'r', encoding='utf-8') as f:
            for ids, line in enumerate(f.readlines()):
                sample = line.strip().split('+')  # 1/1_1_001537 49 90  ego-car hits a crossing pedestrian
                sample1 = sample[0].strip().split(' ')  # 1/1_1_001537 ，49， 90
                word = sample[1].replace('\xa0', ' ')  # ego-car hits a crossing pedestrian
                word.strip()
                data_list.append(sample1[0])  # 1/1_1_001537
                labels.append(int(sample1[0].split('/')[0])-1)
                clips.append([int(sample1[1]), int(sample1[2])])
                texts.append(word)
        return data_list, labels, clips, texts

    def __len__(self):
        return len(self.data_list)

    def get_classes_list(self):
        list_file = os.path.join(self.root_path, 'classes.txt')
        assert os.path.exists(list_file), "%s file does not exist!" % list_file
        self.classes = []
        with open(list_file, 'r', encoding='utf-8') as f:
            for ids, line in enumerate(f.readlines()):
                self.classes.append(line)
        return self.classes

    def read_video(self, video_file, start, end, num_segments):
        video_clip = []
        tmp_repos = []
        video_name = video_file[0].split("/")[-4]+'/'+video_file[0].split("/")[-3]
        # print(video_name, start, end)
        assert end+1-start == num_segments, f"{video_name}' sampling is not {num_segments} frames'"
        for fid in range(start-1, end, self.interval):
            video_data = video_file[fid]
            video_data = Image.open(video_data)
            video_data = self.transforms(video_data)
            video_data = np.asarray(video_data, np.float32)
            video_clip.append(video_data)
        # if len(video_datas) < num_segments:
        #     video_datas += list(video_datas[0] for i in range(num_segments - (end - start)-1))
        frames = [np.asarray(frame) for frame in video_clip]
        frames = torch.as_tensor(np.stack(frames))  # 4D tensor
        if self.phase == 'training':
            tensor_clip = frames  # 8,3,720,1280
            # tensor_clip = tensor_clip.permute(0,1,4,2,3)
        else:
            tensor_clip = torch.unsqueeze(frames, dim=0)
            # tensor_clip = tensor_clip.permute(0,3,1,2)
        return tensor_clip

    def gather_info(self, index):
        accident_id = int(self.data_list[index].split('/')[0])
        video_id = int(self.data_list[index].split('/')[1])
        texts = self.texts[index]
        start, end = self.clips[index]
        # data_info = np.array([accident_id, video_id, start, end], dtype=np.float32)
        data_info = [accident_id, video_id]
        y = torch.tensor(self.labels[index], dtype=torch.int64)
        data_info = torch.tensor(data_info)
        return data_info, y, texts


    def __getitem__(self, index):
        start, end = self.clips[index]
        video_path = os.path.join(self.root_path, self.phase, self.data_list[index], 'images')
        if self.phase == "testing":
            video_path = glob.glob(video_path + '/' + "*.png")
        else:
            video_path = glob.glob(video_path + '/' + "*.jpg")
        video_path = sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0])))
        # for j in range(8):
        #     # imgs=video_data[j].permute(0,2,3,1).numpy()
        #     imgs=video_data
        #     plt.subplot(4,2,j+1)
        #     plt.imshow(imgs)
        # plt.show()
        # exit()
        # c = '/home/ubuntu/CAP-DATA/training/2/2_7_005968/images/'+'000040.jpg'
        # c = Image.open(c)
        # c.show()
        # for i in  range(8):
        #     imgs = video_data[i].transpose(1,2,0)
        #     imgs = Image.fromarray(np.uint8(imgs*255))
        #     imgs.show()
        video_data = self.read_video(video_path, start, end, num_segments=self.num_segments)
        data_info, y, texts = self.gather_info(index)
        return video_data, y, texts

#
# if __name__=="__main__":
#     root = "/home/ubuntu/CAP-DATA"
#     phase = "validation"
#     # model = CAPDATA(root, phase)
# #     # get_data_list,a,b,c=model.get_data_list()
# #     # print(model.__getitem__(10))
# #     from torchvision import transforms
#     from torch.utils.data import DataLoader
# #
# #     # device = torch.device('cuda:0')
# #     num_epochs = 50
# #     transform = transforms.Compose(
# #         [
# #             transforms.ToTensor(),
# #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# #         ]
# #     )
# #     # learning_rate = 0.0001
#     batch_size = 1
# #     shuffle = True
# #     pin_memory = True
#     num_workers = 1
# #     frame_interval = 1
# #     input_shape = [224, 224]
# #     seed = 123
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     val_data = CAPDATA(root, 'validation', interval=1, transform=None,num_segments=8)
#
#
#     val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False,
#                                   num_workers=num_workers, pin_memory=True, drop_last=True)
#
#     a = []
#     for video_data, data_info, y, texts in val_data:
#         # print(video_data.shape, data_info[0:2], texts)
#         # print(y)
#         a.append(y)
# print(len(a))
