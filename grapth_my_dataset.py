import cv2
import json
from torch.utils.data import Dataset
import os
import numpy as np
from einops import rearrange
from PIL import Image
import torch
import glob
import pickle


class CoordinateProcessor:
    def __init__(self):
        self.target_width = 224
        self.target_height = 224
        self.original_width = 1280
        self.original_height = 720

    def process_coordinate(self, json_file):
        coordinate_list = []

        # Scaling factors based on the original and target dimensions
        scale_x = self.target_width / self.original_width
        scale_y = self.target_height / self.original_height

        for i in range(16):
            json_file_path = json_file[i]
            with open(json_file_path, 'rb') as f:
                coordinate_data = json.load(f)
                coordinates = []

                # Loop through all the coordinates in the JSON file
                for key in coordinate_data:
                    for coord in coordinate_data[key]:
                        # Rescale the coordinates
                        x, y = coord
                        rescaled_x = x * scale_x
                        rescaled_y = y * scale_y

                        # Store each coordinate as a list
                        coordinates.append([rescaled_x, rescaled_y])

                # Add the coordinates (list of lists) to the final coordinate list
                coordinate_list.append(coordinates)

        return coordinate_list


class DADA2KS(Dataset):
    def __init__(self, root_path, interval,phase,
                  data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase=phase
        self.data_list, self.c_text,self.p_text,self.pre_text,self.r_texts= self.get_data_list()
        self.target_width = 224
        self.target_height = 224
        self.original_width = 1280
        self.original_height = 720


    def get_data_list(self):
        if self.phase =="train":
            # list_file = os.path.join(self.root_path+"/"+'training_filtered.txt')
            list_file = os.path.join(self.root_path + "/" + 'training_filtered.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs,c_texts,p_texts= [], [],[]

            with open(list_file, 'r',encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split('//')  # e.g.: 1/002 1 0 149 136
                    fileIDs.append(sample[0])
                    c_word = sample[1].replace('\xa0', ' ')
                    c_texts.append(c_word.strip())
                    p_word = sample[2].replace('\xa0', ' ')
                    p_texts.append(p_word.strip())
            return fileIDs,c_texts,p_texts

        if self.phase == "val":
            list_file = os.path.join(self.root_path + "/" + 'demo_text.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, c_texts, p_texts = [], [], []

            with open(list_file, 'r', encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split('//')  # e.g.: 1/002 1 0 149 136
                    fileIDs.append(sample[0])
                    c_word = sample[1].replace('\xa0', ' ')
                    c_texts.append(c_word.strip())
                    p_word = sample[2].replace('\xa0', ' ')
                    p_texts.append(p_word.strip())
            return fileIDs, c_texts, p_texts


    def __len__(self):
        return len(self.data_list)

    def fill_missing_values(self,lst):
        last_valid = None  # 记录最近的非空值
        for i in range(len(lst)):
            if lst[i].numel() == 0 or (lst[i].dim()==0  and lst[i].item()==0):
                if last_valid is not None:
                    lst[i] = last_valid.clone()  # 复制上一个有效值
            else:
                last_valid = lst[i]  # 更新最近的非空值
        return lst


    def pross_video_data(self,video):
         video_datas=[]
         for fid in range(len(video)):
             video_data=video[fid]
             video_data=Image.open(video_data)
             video_data = video_data.resize((224, 224))
             video_data= np.asarray(video_data, np.float32)
             video_datas.append(video_data)

         # guide_image=video_datas[0]
         # guide_image = rearrange(guide_image, 'w h c -> c w h')
         video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
         video_data = rearrange(video_data, 'f w h c -> f c w h')
         return video_data


    def pross_depth_data(self,video):
        video_datas=[]
        for fid in range(16):
            video_data = video[fid]
            img=cv2.imread(video_data,cv2.IMREAD_GRAYSCALE)
            if img.dtype==np.uint16:
                img=img.astype(np.float32) /256.0

            img=cv2.resize(img,(224,224))
            img=np.expand_dims(img,axis=-1)
            video_datas.append(img)
        depth_video = np.array(video_datas, dtype=np.float32)  # 4D tensor
        depth_video = rearrange(depth_video, 'f w h c -> f c w h')
        return depth_video




    def process_relation(self,json_file,json_number_file):
        object_number_list=[]
        relation_number_list=[]
        relations_feature_list = []
        objects_list = []
        map_number_list=[]
        for i in range(16):
            json_file_path = json_file[i]
            json_number_path=json_number_file[i]
            with open(json_number_path, 'rb') as f:
                data =json.load(f)
                if not data['objects']:  # 如果 objects 是空字典
                    object_number_tensor = torch.tensor(0)
                    # mapped_relations_tensor = torch.tensor(0)
                else:
                    # objects = {key.split('_')[0]: value for key, value in data['objects'].items()}
                    # object_keys = list(objects.keys())  # 获取所有键并转换为列表
                    # object_keys_int = [int(key) for key in object_keys]  # 将字符串类型的键转换为整数
                    # object_number_tensor = torch.tensor(object_keys_int)  # 转换为张量
                    object_number_tensor = torch.tensor([int(key.split('_')[0]) for key in data['objects'].keys()])
                    object_keys = list(data['objects'].keys())
                    unique_sorted_objects = sorted(set(object_keys), key=lambda x: (int(x.split('_')[0]), x))
                    object_index_mapping = {obj: idx for idx, obj in enumerate(unique_sorted_objects)}
                    # 处理 relations 部分，若为空则用 torch.tensor(0) 替代
                if not data['relations']:  # 如果 relations 是空列表
                    relation_number_tensor =torch.zeros(1,3,dtype=torch.long)
                    mapped_relations_tensor =torch.zeros(1,3,dtype=torch.long)

                else:
                    relations = [
                        [int(x.split('_')[0]) if isinstance(x, str) and '_' in x else int(x) for x in relation[:3]]
                        for relation in data['relations']
                    ]

                    relations_r = [
                        [relation[:3]]
                        for relation in data['relations']
                    ]

                    mapped_relations = [
                        [[object_index_mapping[x] if x in object_index_mapping else x for x in relation[:2]] + [
                            relation[2]]]
                        for sublist in relations_r for relation in sublist
                    ]
                    relation_number_tensor = torch.tensor(relations)
                    mapped_relations_tensor= torch.tensor(mapped_relations)
                object_number_list.append(object_number_tensor)
                relation_number_list.append(relation_number_tensor)
                map_number_list.append(mapped_relations_tensor)

            with open(json_file_path, 'rb') as f:
                data = pickle.load(f)
                # 使用 .get() 来安全获取数据，如果没有则使用空列表
                objects_data = data.get('objects', [])
                relations_data = data.get('relations', [])
                # 判断对象数据是否为空
                if len(objects_data) > 0:
                    objects = torch.tensor(objects_data)
                else:
                    objects = torch.empty(0)

                    # 判断关系数据是否为空
                if len(relations_data) > 0:
                    relations_features = torch.tensor(relations_data)
                else:
                    relations_features = torch.empty(0)
                # 将数据添加到列表中
                relations_feature_list.append(relations_features)
                objects_list.append(objects)

        objects_list=self.fill_missing_values(objects_list)
        relations_feature_list=self.fill_missing_values(relations_feature_list)
        object_number_list=self.fill_missing_values(object_number_list)
        relation_number_list=self.fill_missing_values(relation_number_list)
        map_number_list=self.fill_missing_values(map_number_list)

        return objects_list,relations_feature_list,object_number_list,relation_number_list,map_number_list














        # def process_coordinate(self, json_file):
    #     coordinate_list = []
    #     # track_ids_list = []  # 用来存储所有的track_ids
    #     for i in range(16):
    #         json_file_path = json_file[i]
    #         with open(json_file_path, 'rb') as f:
    #             coordinate_data = json.load(f)
    #             coordinates = []
    #             # track_ids = []  # 用来存储每个文件的track_ids
    #             # 解析coordinate_data
    #             for key in coordinate_data:
    #                 coordinates.extend(coordinate_data[key])
    #                 # 提取track_id_xx中的xx并加入track_ids
    #                 # match = re.search(r'track_id_(\d+)', key)  # 使用正则表达式提取数字部分
    #                 # if match:
    #                 #     track_ids.append(int(match.group(1)))  # 提取到的数字作为整数添加到track_ids中
    #             # 将坐标转为tensor并堆叠
    #             coordinate_tensor = [torch.tensor(coord) for coord in coordinates if coord is not None]
    #             coordinate_tensors = torch.stack(coordinate_tensor)
    #             # 分别将track_ids和坐标数据加入到对应的列表
    #             # track_ids_list.append(track_ids)
    #             coordinate_list.append(coordinate_tensors)  # 坐标数据单独加入
    #
    #     return coordinate_list

    def process_relationsssss(self, json_file, json_number_file):
        object_number_list = []
        relation_number_list = []
        relations_feature_list = []
        objects_list = []
        object_index_mappings = []  # 存储对象索引映射
        relation_index_mappings = []  # 存储关系索引映射

        for i in range(16):
            json_file_path = json_file[i]
            json_number_path = json_number_file[i]

            with open(json_number_path, 'rb') as f:
                data = json.load(f)

                # 提取对象索引
                if not data['objects']:
                    object_number_tensor = torch.tensor(0)
                    object_index_mapping = {}
                else:
                    object_keys = list(data['objects'].keys())
                    unique_sorted_objects = sorted(set(object_keys), key=lambda x: (int(x.split('_')[0]), x))
                    object_index_mapping = {obj: idx for idx, obj in enumerate(unique_sorted_objects)}
                    object_number_tensor = torch.tensor([int(key.split('_')[0]) for key in object_keys])



                # 处理关系索引
                if not data['relations']:
                    relation_number_tensor = torch.tensor(0)
                    relation_index_mapping = []
                else:
                    relations = [
                        [object_index_mapping[str(x)] if str(x) in object_index_mapping else object_index_mapping[
                            str(x).split('_')[0]]
                         for x in relation[:3]]
                        for relation in data['relations']
                    ]

                    relation_number_tensor = torch.tensor(relations)
                    relation_index_mapping = relations  # 存储索引映射

                object_number_list.append(object_number_tensor)
                relation_number_list.append(relation_number_tensor)
                object_index_mappings.append(object_index_mapping)
                relation_index_mappings.append(relation_index_mapping)

            with open(json_file_path, 'rb') as f:
                data = pickle.load(f)

                objects_data = data.get('objects', [])
                relations_data = data.get('relations', [])

                objects = torch.tensor(objects_data) if len(objects_data) > 0 else torch.empty(0)
                relations_features = torch.tensor(relations_data) if len(relations_data) > 0 else torch.empty(0)

                relations_feature_list.append(relations_features)
                objects_list.append(objects)

        return objects_list, relations_feature_list, object_number_list, relation_number_list, object_index_mappings, relation_index_mappings



    def process_coordinate(self, json_file):
        coordinate_list = []
        # Scaling factors based on the original and target dimensions
        scale_x = self.target_width / self.original_width
        scale_y = self.target_height / self.original_height

        for i in range(16):
            json_file_path = json_file[i]
            with open(json_file_path, 'rb') as f:
                coordinate_data = json.load(f)
                coordinates = []

                # Loop through all the coordinates in the JSON file
                for key in coordinate_data:
                    for coord in coordinate_data[key]:
                        if coord is not None:
                            # Rescale the coordinates
                            x, y = coord
                            rescaled_x = int(x * scale_x)
                            rescaled_y = int(y * scale_y)

                            # Store each coordinate as a list
                            coordinates.append([rescaled_x, rescaled_y])

                # Add the coordinates (list of lists) to the final coordinate list
                coordinate_list.append(coordinates)

        return coordinate_list


    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        tran_video_data=video_file[-16:]
        tran_video_data=self.pross_video_data(tran_video_data)
        return tran_video_data


    def gather_info(self, index):
        accident_id =self.data_list[index]
        c_text=self.c_text[index]
        p_text=self.p_text[index]
        return accident_id,c_text,p_text


    def __getitem__(self, index):
        coordinate_path=os.path.join(r"/media/lotvs/TOSHIBA EXT/Graph_Train/Train_coordinate_new",self.data_list[index])
        coordinate_path=glob.glob(coordinate_path+'/'+"*.json")[5:]
        coordinate_path = sorted(coordinate_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        depth_path=os.path.join(r"/media/lotvs/TOSHIBA EXT/Graph_Train/Train_Videos_Depth",self.data_list[index])
        depth_path = glob.glob(depth_path + '/' + "*.jpg")[5:]
        depth_path= sorted( depth_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))

        mask_path = os.path.join(r"/media/lotvs/TOSHIBA EXT/Graph_Train/Train_Seg_Track_Mask", self.data_list[index])
        mask_path = glob.glob(mask_path+ '/' + "*.png")
        mask_path = sorted(mask_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[0]))
        coordinates= self.process_coordinate(coordinate_path)

        relation_path = os.path.join(self.root_path,self.data_list[index])

        relation_path_number=glob.glob(relation_path+'/'+"*.json")[5:]

        relation_path = glob.glob(relation_path + '/' + "*.pkl")[5:]
        relation_path= sorted(relation_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))

        relation_path_number=sorted(relation_path_number, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        # print("error:",relation_path)
        object_id_clip, relation_clip,id_number,relation_number,map_number_list = self.process_relation(relation_path,relation_path_number)

        accident_id,c_text,p_text= self.gather_info(index)
        video_path=os.path.join(r"/media/lotvs/TOSHIBA EXT/Graph_Train/Train_Videos",self.data_list[index]+"/images")
        video_path = glob.glob( video_path + '/' + "*.jpg")[5:]
        video_path = sorted(video_path , key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        videos=self.read_nomarl_rgbvideo(video_path[-16:])
        depth_video=self.pross_depth_data(depth_path[-16:])
        mask_video = self.pross_depth_data(mask_path[-16:])
        example = {
        "rgb_video":videos / 127.5- 1.0,
        "object_id_clip": object_id_clip,
        "relation_clip": relation_clip,
        "prompt":c_text,
        "p_prompt":p_text,
        "coordinate":coordinates,
        "id_number":id_number,
        "relation_number":relation_number,
        "depth_video":depth_video / 127.5- 1.0,
        "mask_video":mask_video /127.5-1.0,
        "map_relation_number":map_number_list}
        return example



class DADA2KS_Inference(Dataset):
    def __init__(self, root_path, interval,phase,
                  data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase=phase
        self.data_list, self.c_text,self.p_text = self.get_data_list()
        self.target_width = 224
        self.target_height = 224
        self.original_width = 1280
        self.original_height = 720


    def get_data_list(self):
        if self.phase =="train":
            # list_file = os.path.join(self.root_path+"/"+'training_filtered.txt')
            list_file = os.path.join(self.root_path + "/" + 'training_filtered.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs,c_texts,p_texts= [], [],[]

            with open(list_file, 'r',encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split('//')  # e.g.: 1/002 1 0 149 136
                    fileIDs.append(sample[0])
                    c_word = sample[1].replace('\xa0', ' ')
                    c_texts.append(c_word.strip())
                    p_word = sample[2].replace('\xa0', ' ')
                    p_texts.append(p_word.strip())
            return fileIDs,c_texts,p_texts

        if self.phase == "val":
            list_file = os.path.join(self.root_path + "/" + 'demo_text.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, c_texts, p_texts = [], [], []

            with open(list_file, 'r', encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split('//')  # e.g.: 1/002 1 0 149 136
                    fileIDs.append(sample[0])
                    c_word = sample[1].replace('\xa0', ' ')
                    c_texts.append(c_word.strip())
                    p_word = sample[2].replace('\xa0', ' ')
                    p_texts.append(p_word.strip())
            return fileIDs, c_texts, p_texts


    def __len__(self):
        return len(self.data_list)

    def fill_missing_values(self,lst):
        last_valid = None  # 记录最近的非空值
        for i in range(len(lst)):
            if lst[i].numel() == 0 or (lst[i].dim()==0  and lst[i].item()==0):
                if last_valid is not None:
                    lst[i] = last_valid.clone()  # 复制上一个有效值
            else:
                last_valid = lst[i]  # 更新最近的非空值
        return lst


    def pross_video_data(self,video):
         video_datas=[]
         for fid in range(len(video)):
             video_data=video[fid]
             video_data=Image.open(video_data)
             video_data = video_data.resize((224, 224))
             video_data= np.asarray(video_data, np.float32)
             video_datas.append(video_data)

         # guide_image=video_datas[0]
         # guide_image = rearrange(guide_image, 'w h c -> c w h')
         video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
         video_data = rearrange(video_data, 'f w h c -> f c w h')
         return video_data


    def pross_depth_data(self,video):
        video_datas=[]
        for fid in range(16):
            video_data = video[fid]
            img=cv2.imread(video_data,cv2.IMREAD_GRAYSCALE)
            if img.dtype==np.uint16:
                img=img.astype(np.float32) /256.0

            img=cv2.resize(img,(224,224))
            img=np.expand_dims(img,axis=-1)
            video_datas.append(img)
        depth_video = np.array(video_datas, dtype=np.float32)  # 4D tensor
        depth_video = rearrange(depth_video, 'f w h c -> f c w h')
        return depth_video




    def process_relation(self,json_file,json_number_file):
        object_number_list=[]
        relation_number_list=[]
        relations_feature_list = []
        objects_list = []
        map_number_list=[]
        for i in range(16):
            json_file_path = json_file[i]
            json_number_path=json_number_file[i]
            with open(json_number_path, 'rb') as f:
                data =json.load(f)
                if not data['objects']:  # 如果 objects 是空字典
                    object_number_tensor = torch.tensor(0)
                    # mapped_relations_tensor = torch.tensor(0)
                else:
                    # objects = {key.split('_')[0]: value for key, value in data['objects'].items()}
                    # object_keys = list(objects.keys())  # 获取所有键并转换为列表
                    # object_keys_int = [int(key) for key in object_keys]  # 将字符串类型的键转换为整数
                    # object_number_tensor = torch.tensor(object_keys_int)  # 转换为张量
                    object_number_tensor = torch.tensor([int(key.split('_')[0]) for key in data['objects'].keys()])
                    object_keys = list(data['objects'].keys())
                    unique_sorted_objects = sorted(set(object_keys), key=lambda x: (int(x.split('_')[0]), x))
                    object_index_mapping = {obj: idx for idx, obj in enumerate(unique_sorted_objects)}
                    # 处理 relations 部分，若为空则用 torch.tensor(0) 替代
                if not data['relations']:  # 如果 relations 是空列表
                    relation_number_tensor =torch.zeros(1,3,dtype=torch.long)
                    mapped_relations_tensor =torch.zeros(1,3,dtype=torch.long)

                else:
                    relations = [
                        [int(x.split('_')[0]) if isinstance(x, str) and '_' in x else int(x) for x in relation[:3]]
                        for relation in data['relations']
                    ]

                    relations_r = [
                        [relation[:3]]
                        for relation in data['relations']
                    ]

                    mapped_relations = [
                        [[object_index_mapping[x] if x in object_index_mapping else x for x in relation[:2]] + [
                            relation[2]]]
                        for sublist in relations_r for relation in sublist
                    ]
                    relation_number_tensor = torch.tensor(relations)
                    mapped_relations_tensor= torch.tensor(mapped_relations)
                object_number_list.append(object_number_tensor)
                relation_number_list.append(relation_number_tensor)
                map_number_list.append(mapped_relations_tensor)

            with open(json_file_path, 'rb') as f:
                data = pickle.load(f)
                # 使用 .get() 来安全获取数据，如果没有则使用空列表
                objects_data = data.get('objects', [])
                relations_data = data.get('relations', [])
                # 判断对象数据是否为空
                if len(objects_data) > 0:
                    objects = torch.tensor(objects_data)
                else:
                    objects = torch.empty(0)

                    # 判断关系数据是否为空
                if len(relations_data) > 0:
                    relations_features = torch.tensor(relations_data)
                else:
                    relations_features = torch.empty(0)
                # 将数据添加到列表中
                relations_feature_list.append(relations_features)
                objects_list.append(objects)

        objects_list=self.fill_missing_values(objects_list)
        relations_feature_list=self.fill_missing_values(relations_feature_list)
        object_number_list=self.fill_missing_values(object_number_list)
        relation_number_list=self.fill_missing_values(relation_number_list)
        map_number_list=self.fill_missing_values(map_number_list)

        return objects_list,relations_feature_list,object_number_list,relation_number_list,map_number_list














        # def process_coordinate(self, json_file):
    #     coordinate_list = []
    #     # track_ids_list = []  # 用来存储所有的track_ids
    #     for i in range(16):
    #         json_file_path = json_file[i]
    #         with open(json_file_path, 'rb') as f:
    #             coordinate_data = json.load(f)
    #             coordinates = []
    #             # track_ids = []  # 用来存储每个文件的track_ids
    #             # 解析coordinate_data
    #             for key in coordinate_data:
    #                 coordinates.extend(coordinate_data[key])
    #                 # 提取track_id_xx中的xx并加入track_ids
    #                 # match = re.search(r'track_id_(\d+)', key)  # 使用正则表达式提取数字部分
    #                 # if match:
    #                 #     track_ids.append(int(match.group(1)))  # 提取到的数字作为整数添加到track_ids中
    #             # 将坐标转为tensor并堆叠
    #             coordinate_tensor = [torch.tensor(coord) for coord in coordinates if coord is not None]
    #             coordinate_tensors = torch.stack(coordinate_tensor)
    #             # 分别将track_ids和坐标数据加入到对应的列表
    #             # track_ids_list.append(track_ids)
    #             coordinate_list.append(coordinate_tensors)  # 坐标数据单独加入
    #
    #     return coordinate_list

    def process_relationsssss(self, json_file, json_number_file):
        object_number_list = []
        relation_number_list = []
        relations_feature_list = []
        objects_list = []
        object_index_mappings = []  # 存储对象索引映射
        relation_index_mappings = []  # 存储关系索引映射

        for i in range(16):
            json_file_path = json_file[i]
            json_number_path = json_number_file[i]

            with open(json_number_path, 'rb') as f:
                data = json.load(f)

                # 提取对象索引
                if not data['objects']:
                    object_number_tensor = torch.tensor(0)
                    object_index_mapping = {}
                else:
                    object_keys = list(data['objects'].keys())
                    unique_sorted_objects = sorted(set(object_keys), key=lambda x: (int(x.split('_')[0]), x))
                    object_index_mapping = {obj: idx for idx, obj in enumerate(unique_sorted_objects)}
                    object_number_tensor = torch.tensor([int(key.split('_')[0]) for key in object_keys])



                # 处理关系索引
                if not data['relations']:
                    relation_number_tensor = torch.tensor(0)
                    relation_index_mapping = []
                else:
                    relations = [
                        [object_index_mapping[str(x)] if str(x) in object_index_mapping else object_index_mapping[
                            str(x).split('_')[0]]
                         for x in relation[:3]]
                        for relation in data['relations']
                    ]

                    relation_number_tensor = torch.tensor(relations)
                    relation_index_mapping = relations  # 存储索引映射

                object_number_list.append(object_number_tensor)
                relation_number_list.append(relation_number_tensor)
                object_index_mappings.append(object_index_mapping)
                relation_index_mappings.append(relation_index_mapping)

            with open(json_file_path, 'rb') as f:
                data = pickle.load(f)

                objects_data = data.get('objects', [])
                relations_data = data.get('relations', [])

                objects = torch.tensor(objects_data) if len(objects_data) > 0 else torch.empty(0)
                relations_features = torch.tensor(relations_data) if len(relations_data) > 0 else torch.empty(0)

                relations_feature_list.append(relations_features)
                objects_list.append(objects)

        return objects_list, relations_feature_list, object_number_list, relation_number_list, object_index_mappings, relation_index_mappings



    def process_coordinate(self, json_file):
        coordinate_list = []
        # Scaling factors based on the original and target dimensions
        scale_x = self.target_width / self.original_width
        scale_y = self.target_height / self.original_height

        for i in range(16):
            json_file_path = json_file[i]
            with open(json_file_path, 'rb') as f:
                coordinate_data = json.load(f)
                coordinates = []

                # Loop through all the coordinates in the JSON file
                for key in coordinate_data:
                    for coord in coordinate_data[key]:
                        if coord is not None:
                            # Rescale the coordinates
                            x, y = coord
                            rescaled_x = int(x * scale_x)
                            rescaled_y = int(y * scale_y)

                            # Store each coordinate as a list
                            coordinates.append([rescaled_x, rescaled_y])

                # Add the coordinates (list of lists) to the final coordinate list
                coordinate_list.append(coordinates)

        return coordinate_list


    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        tran_video_data=video_file[0:16]
        tran_video_data=self.pross_video_data(tran_video_data)
        return tran_video_data


    def gather_info(self, index):
        accident_id =self.data_list[index]
        c_text=self.c_text[index]
        p_text=self.p_text[index]
        return accident_id,c_text,p_text


    def __getitem__(self, index):
        coordinate_path=os.path.join(r"/media/lotvs/TOSHIBA EXT/Graph_Train/Train_coordinate_new",self.data_list[index])
        coordinate_path=glob.glob(coordinate_path+'/'+"*.json")[5:]
        coordinate_path = sorted(coordinate_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        depth_path=os.path.join(r"/media/lotvs/TOSHIBA EXT/Graph_Train/Train_Videos_Depth",self.data_list[index])
        depth_path = glob.glob(depth_path + '/' + "*.jpg")[5:]
        depth_path= sorted( depth_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))

        mask_path = os.path.join(r"/media/lotvs/TOSHIBA EXT/Graph_Train/Train_Seg_Track_Mask", self.data_list[index])
        mask_path = glob.glob(mask_path+ '/' + "*.png")
        mask_path = sorted(mask_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[0]))
        coordinates= self.process_coordinate(coordinate_path)

        relation_path = os.path.join(self.root_path,self.data_list[index])

        relation_path_number=glob.glob(relation_path+'/'+"*.json")[5:]

        relation_path = glob.glob(relation_path + '/' + "*.pkl")[5:]
        relation_path= sorted(relation_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))

        relation_path_number=sorted(relation_path_number, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        # print("error:",relation_path)
        object_id_clip, relation_clip,id_number,relation_number,map_number_list = self.process_relation(relation_path,relation_path_number)

        accident_id,c_text,p_text= self.gather_info(index)
        video_path=os.path.join(r"/media/lotvs/TOSHIBA EXT/Graph_Train/Train_Videos",self.data_list[index]+"/images")
        video_path = glob.glob( video_path + '/' + "*.jpg")[5:]
        video_path = sorted(video_path , key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        videos=self.read_nomarl_rgbvideo(video_path)
        depth_video=self.pross_depth_data(depth_path)
        mask_video = self.pross_depth_data(mask_path)
        example = {
        "rgb_video":videos / 127.5- 1.0,
        "object_id_clip": object_id_clip,
        "relation_clip": relation_clip,
        "prompt":c_text,
        "p_prompt":p_text,
        "coordinate":coordinates,
        "id_number":id_number,
        "relation_number":relation_number,
        "depth_video":depth_video / 127.5- 1.0,
        "mask_video":mask_video /127.5-1.0,
        "map_relation_number":map_number_list}
        return example









class DADA2KS_Graph_Train(Dataset):
    def __init__(self, root_path, interval,phase,
                  data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase=phase
        self.data_list, self.c_text,self.p_text,self.pre_texts,self.r_texts = self.get_data_list()
        self.target_width = 224
        self.target_height = 224
        self.original_width = 1280
        self.original_height = 720

    #     self.accident_types = [
    #     "ego-car hits a crossing pedestrian",
    #     "ego-car hits a pedestrian",
    #     "ego-car hits a crossing cyclist",
    #     "ego-car hits a cyclist",
    #     "ego-car hits a crossing motorbike",
    #     "ego-car hits a motorbike",
    #     "ego-car hits a crossing truck",
    #     "ego-car hits a truck",
    #     "ego-car is overtaken by a truck",
    #     "ego-car hits a crossing car",
    #     "ego-car hits a car",
    #     "ego-car is overtaken by a car",
    #     "ego-car hits large roadblocks",
    #     "ego-car hits a curb",
    #     "ego-car hits small roadblocks road pothholes",
    #     "ego-car hits trees",
    #     "ego-car hits telegraphpoles",
    #     "ego-car hits other road facilties",
    #     "motorbike hits large roadblocks",
    #     "truck hits large roadblocks",
    #     "car hits large roadblocks",
    #     "motorbike hits a curb",
    #     "truck hits a curb",
    #     "car hits a curb",
    #     "truck hits small roadblocks road pothholes",
    #     "car hits small roadblocks road pothholes",
    #     "truck hits trees",
    #     "car hits trees",
    #     "truck hits telegraphpoles",
    #     "car hits telegraphpoles",
    #     "motorbike hits other road facilities",
    #     "truck hits other road facilities",
    #     "car hits other road facilities",
    #     "motorbike falls down",
    #     "motorbike hits motorbike",
    #     "truck is out of control",
    #     "truck hits truck",
    #     "truck has failure of components",
    #     "car is out of control",
    #     "car hits car",
    #     "car has failure of components",
    #     "truck hits motorbike",
    #     "motorbike scratches truck",
    #     "truck hits car",
    #     "car scratches truck",
    #     "car hits motorbike",
    #     "motorbike scratches car",
    #     "motorbike hits pedestrian",
    #     "motorbike hits cyclist",
    #     "truck hits pedestrian",
    #     "truck hits cyclist",
    #     "car hits pedestrian",
    #     "car hits cyclist",
    #     "pedestrian falls down",
    #     "cyclist falls down",
    #     "cyclist hits cyclist",
    #     "ego-car is out of control",
    #     "ego-car has failure of components",
    # ]

        # # 构造提示列表
        # type_hint = "\n".join([f"- {t}" for t in self.accident_types])
        # self.type_prompt = f"The accident type should be chosen from the following list:\n{type_hint}"

    def get_data_list(self):
        if self.phase =="train":
            # list_file = os.path.join(self.root_path+"/"+'training_filtered.txt')
            list_file = os.path.join(self.root_path + "/" + 'train_v_lava.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs,c_texts,p_texts,pre_texts,r_texts= [], [],[],[],[]

            with open(list_file, 'r',encoding='utf-8') as f:
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split('//')  # e.g.: 1/002 1 0 149 136
                    fileIDs.append(sample[0])
                    c_word = sample[1].replace('\xa0', ' ')
                    c_texts.append(c_word.strip())
                    a_word = sample[3].replace('\xa0', ' ')
                    p_texts.append(a_word.strip())
                    pre_word = sample[2].replace('\xa0', ' ')
                    pre_texts.append(pre_word.strip())
                    r_word = sample[4].replace('\xa0', ' ')
                    r_texts.append(r_word.strip())
            return fileIDs,c_texts,p_texts,pre_texts,r_texts

        if self.phase == "val":
            list_file = os.path.join(self.root_path + "/" + 'demo_text.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, c_texts, p_texts = [], [], []

            with open(list_file, 'r', encoding='utf-8') as f:
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split('//')  # e.g.: 1/002 1 0 149 136
                    fileIDs.append(sample[0])
                    c_word = sample[1].replace('\xa0', ' ')
                    c_texts.append(c_word.strip())
                    p_word = sample[2].replace('\xa0', ' ')
                    p_texts.append(p_word.strip())
            return fileIDs, c_texts, p_texts


    def __len__(self):
        return len(self.data_list)

    def fill_missing_values(self,lst):
        last_valid = None 
        for i in range(len(lst)):
            if lst[i].numel() == 0 or (lst[i].dim()==0  and lst[i].item()==0):
                if last_valid is not None:
                    lst[i] = last_valid.clone()  
            else:
                last_valid = lst[i]  


    def norma_box(self,box_list):
        normalized_box_list = []
        for box_tensor in box_list:
            x1, y1, x2, y2 = box_tensor[:, 0], box_tensor[:, 1], box_tensor[:, 2], box_tensor[:, 3]
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            x_center /= 1280
            w=w.float() / 1280
            y_center /= 720
            h=h.float() / 720
            norm_box_tensor = torch.stack([x_center, y_center, w, h], dim=1)  # shape: (N_i, 4)
            normalized_box_list.append(norm_box_tensor)
        return  normalized_box_list


    def pross_video_data(self,video):
         video_datas=[]
         for fid in range(len(video)):
             video_data=video[fid]
             video_data=Image.open(video_data)
             video_data = video_data.resize((224, 224))
             video_data= np.asarray(video_data, np.float32)
             video_datas.append(video_data)
         video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
         video_data = rearrange(video_data, 'f w h c -> f c w h')
         return video_data

    def pross_depth_data(self, video):
        video_datas = []
        for fid in range(len(video)):  
            video_data = video[fid]
            img = cv2.imread(video_data, cv2.IMREAD_GRAYSCALE)
            if img.dtype == np.uint16:
                img = img.astype(np.float32) 
            else:
                img = img.astype(np.float32)
            img = cv2.resize(img, dsize=(224, 224))
            img = np.expand_dims(img, axis=-1)
            video_datas.append(img)
        while len(video_datas) < 16:
            video_datas.append(video_datas[-1].copy())
        depth_video = np.array(video_datas, dtype=np.float32)  # shape: (T, H, W, 1)
        depth_video = rearrange(depth_video, 'f h w c -> f c w h')  # rearrange to match expected format
        return depth_video

    def encode_objects_and_triples(self,object_keys, relations_r):
        """
        object_keys: list of str (e.g., ['1', '6', '6_1', '6_2'])
        relations_r: list of triple-like lists, e.g., [['1', 4, '6']]
        Returns:
            - object_key_to_index: dict {str: int}
            - triples_index: Tensor [N_rel, 3], with source/destination as index
        """
        object_key_to_index = {key: idx for idx, key in enumerate(object_keys)}
        triples_index = []
        for relation in relations_r:
            src_key, rel_type, dst_key = relation
            src_idx = object_key_to_index[src_key]
            dst_idx = object_key_to_index[dst_key]
            triples_index.append([src_idx, rel_type, dst_idx])
        return object_key_to_index, torch.tensor(triples_index, dtype=torch.long)

    def process_relation(self,json_file):
        object_number_list = []
        relation_number_list = []
        relation_texts = []
        object_texts = []
        box_list = []
        angle_list = []
        gt_labels=[]
        for i in range(16):
            json_number_path = json_file[i]
            with open(json_number_path, 'rb') as f:
                data = json.load(f)

                if not data['objects']:
                    object_number_tensor = torch.zeros(1,dtype=torch.long)
                    boxes_tensor = torch.zeros(1, 4)
                    object_text = ["the object is {}"]
                    object_keys = []
                else:
                    object_keys = list(data['objects'].keys())
                    object_number_tensor = torch.tensor([int(k.split('_')[0]) for k in object_keys])
                    object_text = [data['objects'][k] for k in object_keys]
                    boxes_tensor = torch.stack([torch.tensor(data['coordinates'][k]) for k in object_keys])

                # Parse relations if present
                if not data['relations']:
                    relation_number_tensor = torch.zeros(1, 3, dtype=torch.long)
                    gt_relation_labels = torch.zeros(1, 3, dtype=torch.long)
                    angles_tensor =torch.zeros(1,dtype=torch.long)
                    r_text = ["the relationship is {}"]
                else:
                    # keys for id conversion
                    relations_raw = [r[:3] for r in data['relations']]  # [['6', 0, '6_1']]
                    gt_relation_labels=[[int(str(item).split('_')[0]) for item in r[:3]] for r in data['relations']]
                    r_text = [r[-2] for r in data['relations']]
                    angles_tensor = torch.tensor([r[-1] for r in data['relations']])

                    _, relation_number_tensor = self.encode_objects_and_triples(object_keys, relations_raw)

                object_number_list.append(object_number_tensor)
                box_list.append(boxes_tensor)
                object_texts.append(object_text)
                relation_number_list.append(relation_number_tensor)
                angle_list.append(angles_tensor)
                relation_texts.append(r_text)
                gt_labels.append(gt_relation_labels)
        # Fill missing frames
        def fill_missing_values(lst):
            last_valid = None
            for i in range(len(lst)):
                if isinstance(lst[i], torch.Tensor) and (
                        lst[i].numel() == 0 or (lst[i].dim() == 0 and lst[i].item() == 0)):
                    if last_valid is not None:
                        lst[i] = last_valid.clone()
                else:
                    last_valid = lst[i]
            return lst

        object_number_list = fill_missing_values(object_number_list)
        relation_number_list = fill_missing_values(relation_number_list)
        box_list = fill_missing_values(box_list)
        box_list=self.norma_box(box_list)




        angle_list = fill_missing_values(angle_list)
        gt_labels=fill_missing_values(  gt_labels)

        return object_number_list, relation_number_list, relation_texts, object_texts, box_list, angle_list,gt_labels
   

    def process_relationsssss(self, json_file, json_number_file):
        object_number_list = []
        relation_number_list = []
        relations_feature_list = []
        objects_list = []
        object_index_mappings = []  
        relation_index_mappings = [] 

        for i in range(16):
            json_file_path = json_file[i]
            json_number_path = json_number_file[i]

            with open(json_number_path, 'rb') as f:
                data = json.load(f)

             
                if not data['objects']:
                    object_number_tensor = torch.tensor(0)
                    object_index_mapping = {}
                else:
                    object_keys = list(data['objects'].keys())
                    unique_sorted_objects = sorted(set(object_keys), key=lambda x: (int(x.split('_')[0]), x))
                    object_index_mapping = {obj: idx for idx, obj in enumerate(unique_sorted_objects)}
                    object_number_tensor = torch.tensor([int(key.split('_')[0]) for key in object_keys])


                if not data['relations']:
                    relation_number_tensor = torch.tensor(0)
                    relation_index_mapping = []
                else:
                    relations = [
                        [object_index_mapping[str(x)] if str(x) in object_index_mapping else object_index_mapping[
                            str(x).split('_')[0]]
                         for x in relation[:3]]
                        for relation in data['relations']
                    ]

                    relation_number_tensor = torch.tensor(relations)
                    relation_index_mapping = relations 

                object_number_list.append(object_number_tensor)
                relation_number_list.append(relation_number_tensor)
                object_index_mappings.append(object_index_mapping)
                relation_index_mappings.append(relation_index_mapping)

            with open(json_file_path, 'rb') as f:
                data = pickle.load(f)

                objects_data = data.get('objects', [])
                relations_data = data.get('relations', [])

                objects = torch.tensor(objects_data) if len(objects_data) > 0 else torch.empty(0)
                relations_features = torch.tensor(relations_data) if len(relations_data) > 0 else torch.empty(0)

                relations_feature_list.append(relations_features)
                objects_list.append(objects)

        return objects_list, relations_feature_list, object_number_list, relation_number_list, object_index_mappings, relation_index_mappings



    def process_coordinate(self, json_file):
        coordinate_list = []
        # Scaling factors based on the original and target dimensions
        scale_x = self.target_width / self.original_width
        scale_y = self.target_height / self.original_height

        for i in range(16):
            json_file_path = json_file[i]
            with open(json_file_path, 'rb') as f:
                coordinate_data = json.load(f)
                coordinates = []

                # Loop through all the coordinates in the JSON file
                for key in coordinate_data:
                    for coord in coordinate_data[key]:
                        if coord is not None:
                            # Rescale the coordinates
                            x, y = coord
                            rescaled_x = int(x * scale_x)
                            rescaled_y = int(y * scale_y)

                            # Store each coordinate as a list
                            coordinates.append([rescaled_x, rescaled_y])

                # Add the coordinates (list of lists) to the final coordinate list
                coordinate_list.append(coordinates)

        return coordinate_list


    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        # tran_video_data=video_file[-16:]
        tran_video_data=self.pross_video_data( video_file)
        return tran_video_data


    def gather_info(self, index):
        accident_id =self.data_list[index]
        c_text=self.c_text[index]
        p_text=self.p_text[index]
        pre_text=self.pre_texts[index]
        r_text=self.r_texts[index]
        return accident_id,c_text,p_text,pre_text,r_text


    def __getitem__(self, index):
        relation_path = os.path.join(self.root_path, self.data_list[index])
        relation_path = glob.glob(relation_path + '/' + "*.json")
        relation_path = sorted(relation_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        depth_path = os.path.join(r"./Graph_Train/Train_Videos_Depth",
                                  self.data_list[index])
        depth_path = glob.glob(depth_path + '/' + "*.jpg")
        depth_path = sorted(depth_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))

        mask_path = os.path.join(r"./Graph_Train/Train_Seg_Track_Mask_New",
                                 self.data_list[index])
        mask_path = glob.glob(mask_path + '/' + "*.png")
        mask_path = sorted(mask_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[0]))
        video_path = os.path.join(r"./Graph_Train/Train_Videos",
                                  self.data_list[index] + "/images")
        video_path = glob.glob(video_path + '/*.jpg')
        video_path = sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        normal_idx = list(range(16))
        # normal_relation_path = [relation_path[i] for i in normal_idx]
        # normal_depth_path = [depth_path[i] for i in normal_idx]
        # normal_mask_path = [mask_path[i] for i in normal_idx]
        normal_video_path = [video_path[i] for i in normal_idx]

        normal_rgb_video = self.read_nomarl_rgbvideo(normal_video_path)
        # normal_depth_video = self.pross_depth_data(normal_depth_path)
        # normal_mask_video = self.pross_depth_data(normal_mask_path )
        # initial_obj_ids, initial_relations, initial_rel_texts, initial_obj_texts, initial_boxes, initial_angles, _ = \
        #     self.process_relation(normal_relation_path)
        accident_id, lava_c_text, lava_p_text,lava_pre_text,r_text= self.gather_info(index)

        if "ASSISTANT:" in lava_p_text:
            user_part, assistant_part = lava_p_text.split("ASSISTANT:")
            user_part = user_part.strip()
            assistant = assistant_part.split(" ")[1].strip()
            answer = f"The type of this predicted accident video is a {assistant} accident"
            lava_a_text = f"{user_part} ASSISTANT:{answer}"
        if "ASSISTANT:" in lava_c_text:
            user_part, assistant_part = lava_p_text.split("ASSISTANT:")
            user_part1, assistant_part1=lava_c_text.split("ASSISTANT:")

            a_c_text = lava_c_text.split("ASSISTANT:")[1].strip()
            c_text = a_c_text.split(",")[0].strip()
            a_text = assistant_part.strip()
            lava_c_text = f"{user_part} ASSISTANT:{r_text}"
            u="USER: <video> What is the reason and category of this interpolated accident video?"
            lava_ac_text = f"{u} ASSISTANT:{a_c_text}"
            lava_re_text = f"{user_part1} ASSISTANT:{c_text}"

        target_len = 32
        def pad_to_32(path_list):
            if len(path_list) < target_len:
                path_list.extend([path_list[-1]] * (target_len - len(path_list)))
            return path_list
    
        relation_path = pad_to_32(relation_path)
        depth_path = pad_to_32(depth_path)
        mask_path = pad_to_32(mask_path)
        video_path = pad_to_32(video_path)
      
        sampled_idx = [target_len - 1 - 2 * i for i in range(16)]
        sampled_idx = sampled_idx[::-1] 
  
        selected_relation_path = [relation_path[i] for i in sampled_idx]
        selected_depth_path = [depth_path[i] for i in sampled_idx]
        selected_mask_path = [mask_path[i] for i in sampled_idx]
        selected_video_path = [video_path[i] for i in sampled_idx]
        videos = self.read_nomarl_rgbvideo( selected_video_path )
        depth_video = self.pross_depth_data(selected_depth_path)
        mask_video = self.pross_depth_data( selected_mask_path )
        object_number_list, relation_number_list, relation_texts, object_texts, box_list, angle_list, gt_labels = self.process_relation(
            selected_relation_path)
        example = {
            "rgb_video": videos / 127.5 - 1.0,
            "depth_video": depth_video / 127.5 - 1.0,
            "mask_video": mask_video / 255,
            "object_id": object_number_list,
            "object_text": object_texts,
            "relation_number": relation_number_list,
            "relation_text": relation_texts,
            "boxes": box_list,
            "angles": angle_list,
            "c_prompt": c_text,
            "a_prompt": a_text,
            "ac_prompt": a_c_text,
            "lava_a_text": lava_a_text,
            "lava_r_text": lava_c_text,
            "lava_ac_text": lava_ac_text,
            "lava_p_text": lava_pre_text,
            "lava_re_text": lava_re_text,
            "gt_labels": gt_labels,
            "n_rgb_video":normal_rgb_video / 127.5 - 1.0,
            # "n_depth_video" =normal_depth_path / 127.5 - 1.0,
            # "n_mask_video" = self.pross_depth_data(normal_mask_path)


        }
        return example




class DADA2KS_Graph_Inference(Dataset):
    def __init__(self, root_path, interval,phase,
                  data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase=phase
        self.data_list, self.c_text,self.p_text,self.a_text= self.get_data_list()
        self.target_width = 224
        self.target_height = 224
        self.original_width = 1280
        self.original_height = 720


    def get_data_list(self):
        if self.phase =="train":
            # list_file = os.path.join(self.root_path+"/"+'training_filtered.txt')
            list_file = os.path.join(self.root_path + "/" + 'train_v_lava.txt')

            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs,c_texts,p_texts= [], [],[]

            with open(list_file, 'r',encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split('//')  # e.g.: 1/002 1 0 149 136
                    fileIDs.append(sample[0])
                    c_word = sample[1].replace('\xa0', ' ')
                    c_texts.append(c_word.strip())
                    a_word = sample[3].replace('\xa0', ' ')
                    p_texts.append(a_word.strip())
            return fileIDs,c_texts,p_texts

        if self.phase == "val":
            list_file = os.path.join(self.root_path + "/" + 'test_lava.txt')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, c_texts, p_texts,a_texts = [], [], [],[]

            with open(list_file, 'r', encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split('//')  # e.g.: 1/002 1 0 149 136
                    fileIDs.append(sample[0])
                    c_word = sample[1].replace('\xa0', ' ')
                    c_texts.append(c_word.strip())
                    p_word = sample[2].replace('\xa0', ' ')
                    p_texts.append(p_word.strip())

                    a_word = sample[3].replace('\xa0', ' ')
                    a_texts.append(a_word.strip())

            return fileIDs, c_texts, p_texts, a_texts


    def __len__(self):
        return len(self.data_list)

    def fill_missing_values(self,lst):
        last_valid = None  # 记录最近的非空值
        for i in range(len(lst)):
            if lst[i].numel() == 0 or (lst[i].dim()==0  and lst[i].item()==0):
                if last_valid is not None:
                    lst[i] = last_valid.clone()  # 复制上一个有效值
            else:
                last_valid = lst[i]  # 更新最近的非空值
        return lst


    def norma_box(self,box_list):
        normalized_box_list = []

        for box_tensor in box_list:
            # 原始坐标
            x1, y1, x2, y2 = box_tensor[:, 0], box_tensor[:, 1], box_tensor[:, 2], box_tensor[:, 3]

            # 转换为中心坐标形式
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1

            # 归一化
            x_center /= 1280
            w=w.float() / 1280
            y_center /= 720
            h=h.float() / 720

            # 拼接为 (x_center, y_center, w, h)
            norm_box_tensor = torch.stack([x_center, y_center, w, h], dim=1)  # shape: (N_i, 4)

            normalized_box_list.append(norm_box_tensor)
        return  normalized_box_list


    def pross_video_data(self,video):
         video_datas=[]
         for fid in range(len(video)):
             video_data=video[fid]
             video_data=Image.open(video_data)
             video_data = video_data.resize((224, 224))
             video_data= np.asarray(video_data, np.float32)
             video_datas.append(video_data)

         # guide_image=video_datas[0]
         # guide_image = rearrange(guide_image, 'w h c -> c w h')
         video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
         video_data = rearrange(video_data, 'f w h c -> f c w h')
         return video_data

    def pross_depth_data(self, video):
        video_datas = []

        for fid in range(len(video)):  # 改为动态长度
            video_data = video[fid]
            img = cv2.imread(video_data, cv2.IMREAD_GRAYSCALE)

            if img.dtype == np.uint16:
                img = img.astype(np.float32) 
            else:
                img = img.astype(np.float32) 

            img = cv2.resize(img, dsize=(224, 224))
            img = np.expand_dims(img, axis=-1)
            video_datas.append(img)

        # 如果不足16帧，用最后一帧补齐
        while len(video_datas) < 16:
            video_datas.append(video_datas[-1].copy())

        # 转成 numpy 数组并变换维度
        depth_video = np.array(video_datas, dtype=np.float32)  # shape: (T, H, W, 1)
        depth_video = rearrange(depth_video, 'f h w c -> f c w h')  # rearrange to match expected format
        return depth_video



    #
    # def process_relation(self,json_file):
    #     object_number_list=[]
    #     relation_number_list= []
    #     relation_texts=  []
    #     object_texts  =   []
    #     # map_number_list=[]
    #     box_list=[]
    #     angle_list=[]
    #
    #     for i in range(16):
    #         json_number_path = json_file[i]
    #         with open(json_number_path, 'rb') as f:
    #             data =json.load(f)
    #             if not data['objects']:  # 如果 objects 是空字典
    #                 object_number_tensor = torch.tensor(0)
    #                 boxes_tensor= torch.zeros(1,4)
    #                 object_text="the object is {}"
    #
    #             else:
    #                 object_number_tensor = torch.tensor([int(key.split('_')[0]) for key in data['objects'].keys()])
    #                 object_keys = list(data['objects'].keys())
    #                 object_text=list([data['objects'][key] for key in object_keys ])
    #
    #                 # unique_sorted_objects = sorted(set(object_keys), key=lambda x: (int(x.split('_')[0]), x))
    #                 # object_index_mapping = {obj: idx for idx, obj in enumerate(unique_sorted_objects)}
    #                 boxes=list(data['coordinates'].keys())
    #                 boxes_tensor=torch.stack(list(torch.tensor([data['coordinates'][key] for key in boxes ])))
    #             object_texts.append(object_text)
    #             box_list.append(boxes_tensor)
    #                 # 处理 relations 部分，若为空则用 torch.tensor(0) 替代
    #             if not data['relations']:  # 如果 relations 是空列表
    #
    #                 relation_number_tensor =torch.zeros(1,3,dtype=torch.long)
    #                 angle=int(0)
    #                 r_text="the relationship is {}"
    #
    #
    #                 # mapped_relations_tensor =torch.zeros(1,3,dtype=torch.long)
    #             else:
    #                 relations = [
    #                     [int(x.split('_')[0]) if isinstance(x, str) and '_' in x else int(x) for x in relation[:3]]
    #                     for relation in data['relations']
    #                 ]
    #
    #                 r_text = [
    #                     [x for x in relation[-2:-1]]
    #                     for relation in data['relations']
    #                 ]
    #
    #                 angle = [
    #                     [x for x in relation[-1:]]
    #                     for relation in data['relations']
    #                 ]
    #
    #                 relations_r = [
    #                     [relation[:3]]
    #                     for relation in data['relations']
    #                 ]
    #
    #                 # mapped_relations = [
    #                 #     [[object_index_mapping[x] if x in object_index_mapping else x for x in relation[:2]] + [
    #                 #         relation[2]]]
    #                 #     for sublist in relations_r for relation in sublist
    #                 # ]
    #                 relation_number_tensor = torch.tensor(relations)
    #                 # mapped_relations_tensor= torch.tensor(mapped_relations)
    #             object_number_list.append(object_number_tensor)
    #             relation_number_list.append(relation_number_tensor)
    #             angle_list.append(torch.tensor(angle).squeeze(1))
    #             relation_texts.append(r_text)
    #
    #             # map_number_list.append(mapped_relations_tensor)
    #
    #     object_number_list =self.fill_missing_values(object_number_list )
    #     relation_number_list=self.fill_missing_values(relation_number_list)
    #     # relation_texts=self.fill_missing_values(relation_texts)
    #     # object_texts=self.fill_missing_values(object_texts)
    #     # map_number_list=self.fill_missing_values(map_number_list)
    #     box_list=self.fill_missing_values(box_list)
    #     # angle_list=self.fill_missing_values(angle_list)
    #
    #     return object_number_list,relation_number_list, relation_texts,object_texts,box_list,angle_list

    def encode_objects_and_triples(self,object_keys, relations_r):
        """
        object_keys: list of str (e.g., ['1', '6', '6_1', '6_2'])
        relations_r: list of triple-like lists, e.g., [['1', 4, '6']]
        Returns:
            - object_key_to_index: dict {str: int}
            - triples_index: Tensor [N_rel, 3], with source/destination as index
        """
        object_key_to_index = {key: idx for idx, key in enumerate(object_keys)}
        triples_index = []
        for relation in relations_r:
            src_key, rel_type, dst_key = relation
            src_idx = object_key_to_index[src_key]
            dst_idx = object_key_to_index[dst_key]
            triples_index.append([src_idx, rel_type, dst_idx])
        return object_key_to_index, torch.tensor(triples_index, dtype=torch.long)

    def process_relation(self,json_file):
        object_number_list = []
        relation_number_list = []
        relation_texts = []
        object_texts = []
        box_list = []
        angle_list = []
        gt_labels=[]
        for i in range(16):
            json_number_path = json_file[i]
            with open(json_number_path, 'rb') as f:
                data = json.load(f)

                if not data['objects']:
                    object_number_tensor = torch.zeros(1,dtype=torch.long)
                    boxes_tensor = torch.zeros(1, 4)
                    object_text = ["the object is {}"]
                    object_keys = []
                else:
                    object_keys = list(data['objects'].keys())
                    object_number_tensor = torch.tensor([int(k.split('_')[0]) for k in object_keys])
                    object_text = [data['objects'][k] for k in object_keys]
                    boxes_tensor = torch.stack([torch.tensor(data['coordinates'][k]) for k in object_keys])

                # Parse relations if present
                if not data['relations']:
                    relation_number_tensor = torch.zeros(1, 3, dtype=torch.long)
                    gt_relation_labels = torch.zeros(1, 3, dtype=torch.long)
                    angles_tensor =torch.zeros(1,dtype=torch.long)
                    r_text = ["the relationship is {}"]
                else:
                    # keys for id conversion
                    relations_raw = [r[:3] for r in data['relations']]  # [['6', 0, '6_1']]
                    gt_relation_labels=[[int(str(item).split('_')[0]) for item in r[:3]] for r in data['relations']]
                    r_text = [r[-2] for r in data['relations']]
                    angles_tensor = torch.tensor([r[-1] for r in data['relations']])

                    _, relation_number_tensor = self.encode_objects_and_triples(object_keys, relations_raw)

                object_number_list.append(object_number_tensor)
                box_list.append(boxes_tensor)
                object_texts.append(object_text)
                relation_number_list.append(relation_number_tensor)
                angle_list.append(angles_tensor)
                relation_texts.append(r_text)
                gt_labels.append(gt_relation_labels)
        # Fill missing frames
        def fill_missing_values(lst):
            last_valid = None
            for i in range(len(lst)):
                if isinstance(lst[i], torch.Tensor) and (
                        lst[i].numel() == 0 or (lst[i].dim() == 0 and lst[i].item() == 0)):
                    if last_valid is not None:
                        lst[i] = last_valid.clone()
                else:
                    last_valid = lst[i]
            return lst

        object_number_list = fill_missing_values(object_number_list)
        relation_number_list = fill_missing_values(relation_number_list)
        box_list = fill_missing_values(box_list)
        box_list=self.norma_box(box_list)




        angle_list = fill_missing_values(angle_list)
        gt_labels=fill_missing_values(  gt_labels)

        return object_number_list, relation_number_list, relation_texts, object_texts, box_list, angle_list,gt_labels
    # def process_coordinate(self, json_file):
    #     coordinate_list = []
    #     # track_ids_list = []  # 用来存储所有的track_ids
    #     for i in range(16):
    #         json_file_path = json_file[i]
    #         with open(json_file_path, 'rb') as f:
    #             coordinate_data = json.load(f)
    #             coordinates = []
    #             # track_ids = []  # 用来存储每个文件的track_ids
    #             # 解析coordinate_data
    #             for key in coordinate_data:
    #                 coordinates.extend(coordinate_data[key])
    #                 # 提取track_id_xx中的xx并加入track_ids
    #                 # match = re.search(r'track_id_(\d+)', key)  # 使用正则表达式提取数字部分
    #                 # if match:
    #                 #     track_ids.append(int(match.group(1)))  # 提取到的数字作为整数添加到track_ids中
    #             # 将坐标转为tensor并堆叠
    #             coordinate_tensor = [torch.tensor(coord) for coord in coordinates if coord is not None]
    #             coordinate_tensors = torch.stack(coordinate_tensor)
    #             # 分别将track_ids和坐标数据加入到对应的列表
    #             # track_ids_list.append(track_ids)
    #             coordinate_list.append(coordinate_tensors)  # 坐标数据单独加入
    #
    #     return coordinate_list

    def process_relationsssss(self, json_file, json_number_file):
        object_number_list = []
        relation_number_list = []
        relations_feature_list = []
        objects_list = []
        object_index_mappings = []  # 存储对象索引映射
        relation_index_mappings = []  # 存储关系索引映射

        for i in range(16):
            json_file_path = json_file[i]
            json_number_path = json_number_file[i]

            with open(json_number_path, 'rb') as f:
                data = json.load(f)

                # 提取对象索引
                if not data['objects']:
                    object_number_tensor = torch.tensor(0)
                    object_index_mapping = {}
                else:
                    object_keys = list(data['objects'].keys())
                    unique_sorted_objects = sorted(set(object_keys), key=lambda x: (int(x.split('_')[0]), x))
                    object_index_mapping = {obj: idx for idx, obj in enumerate(unique_sorted_objects)}
                    object_number_tensor = torch.tensor([int(key.split('_')[0]) for key in object_keys])



                # 处理关系索引
                if not data['relations']:
                    relation_number_tensor = torch.tensor(0)
                    relation_index_mapping = []
                else:
                    relations = [
                        [object_index_mapping[str(x)] if str(x) in object_index_mapping else object_index_mapping[
                            str(x).split('_')[0]]
                         for x in relation[:3]]
                        for relation in data['relations']
                    ]

                    relation_number_tensor = torch.tensor(relations)
                    relation_index_mapping = relations  # 存储索引映射

                object_number_list.append(object_number_tensor)
                relation_number_list.append(relation_number_tensor)
                object_index_mappings.append(object_index_mapping)
                relation_index_mappings.append(relation_index_mapping)

            with open(json_file_path, 'rb') as f:
                data = pickle.load(f)

                objects_data = data.get('objects', [])
                relations_data = data.get('relations', [])

                objects = torch.tensor(objects_data) if len(objects_data) > 0 else torch.empty(0)
                relations_features = torch.tensor(relations_data) if len(relations_data) > 0 else torch.empty(0)

                relations_feature_list.append(relations_features)
                objects_list.append(objects)

        return objects_list, relations_feature_list, object_number_list, relation_number_list, object_index_mappings, relation_index_mappings

    def process_coordinate(self, json_file):
        coordinate_list = []
        # Scaling factors based on the original and target dimensions
        scale_x = self.target_width / self.original_width
        scale_y = self.target_height / self.original_height

        for i in range(16):
            json_file_path = json_file[i]
            with open(json_file_path, 'rb') as f:
                coordinate_data = json.load(f)
                coordinates = []

                # Loop through all the coordinates in the JSON file
                for key in coordinate_data:
                    for coord in coordinate_data[key]:
                        if coord is not None:
                            # Rescale the coordinates
                            x, y = coord
                            rescaled_x = int(x * scale_x)
                            rescaled_y = int(y * scale_y)

                            # Store each coordinate as a list
                            coordinates.append([rescaled_x, rescaled_y])

                # Add the coordinates (list of lists) to the final coordinate list
                coordinate_list.append(coordinates)

        return coordinate_list


    def read_nomarl_rgbvideo(self, video_file):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        tran_video_data=video_file[-16:]
        tran_video_data=self.pross_video_data(tran_video_data)
        return tran_video_data


    def gather_info(self, index):
        accident_id =self.data_list[index]
        c_text=self.c_text[index]
        p_text=self.p_text[index]
        a_text=self.a_text[index]
        return accident_id,c_text,p_text,a_text


    def __getitem__(self, index):
        relation_path = os.path.join(self.root_path,self.data_list[index])
        save_name=self.data_list[index]
        relation_path=glob.glob(relation_path+'/'+"*.json")
        relation_path=sorted(relation_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        # object_number_list, relation_number_list, relation_texts, object_texts,box_list, angle_list,gt_labels = self.process_relation(relation_path)
        # accident_id,c_text,p_text= self.gather_info(index)

        # coordinate_path = os.path.join(r"/media/lotvs/TOSHIBA EXT1/Graph_Train/Train_coordinate_new",
        #                                self.data_list[index])
        # coordinate_path = glob.glob(coordinate_path + '/' + "*.json")
        # coordinate_path = sorted(coordinate_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        target_len = 32
        def pad_to_32(path_list, target_len=32):
            if len(path_list) == 0:
                raise ValueError("pad_to_32 received an empty path_list")

            if len(path_list) < target_len:
                path_list.extend([path_list[-1]] * (target_len - len(path_list)))
            return path_list
        # 单独对每个模态做补齐
        depth_path = os.path.join(r"/home/lotvs/Code/Test_Data/Test_Depth", self.data_list[index])
        depth_path = glob.glob(depth_path + '/' + "*.jpg")
        depth_path = sorted(depth_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        mask_path = os.path.join(r"/home/lotvs/Code/Test_Data/Test_Mask", self.data_list[index])
        mask_path = glob.glob(mask_path + '/' + "*.png")
        mask_path = sorted(mask_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[0]))
        # coordinates = self.process_coordinate(coordinate_path)
        accident_id, lava_c_text, lava_p_text,lava_a_text= self.gather_info(index)
        if "ASSISTANT:" in lava_c_text:
            c_text = lava_c_text.split("ASSISTANT:")[1].strip()
            a_text=lava_p_text.split("ASSISTANT:")[1].strip()
            aa_text=lava_a_text.split("ASSISTANT:")[1].strip()
        video_path = os.path.join(r"/home/lotvs/Code/Test_Data/Test_Video",
                                  self.data_list[index] + "/images")
        video_path = glob.glob(video_path + '/*.jpg')
        video_path = sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        
        mask_path = pad_to_32(mask_path)
        video_path = pad_to_32(video_path)
        relation_path = pad_to_32(relation_path)
        depth_path = pad_to_32(depth_path)
        mask_path = pad_to_32(mask_path)
        video_path = pad_to_32(video_path)
        # sampled_idx = [target_len - 1 - 2 * i for i in range(16)]
        # sampled_idx = sampled_idx[::-1]  
        # # # 取出对应帧
        # selected_relation_path = [relation_path[i] for i in sampled_idx]
        # selected_depth_path = [depth_path[i] for i in sampled_idx]
        # selected_mask_path = [mask_path[i] for i in sampled_idx]
        # selected_video_path = [video_path[i] for i in sampled_idx]
        # #
        start_index=os.path.basename(video_path[-16:][0]).split(".")[0]
        end_index = os.path.basename(video_path[-16:][-1]).split(".")[0]
        #if predict,use video_path[-16:];depth_path[-16:],mask_path[-16:],relation_path[-16:]
        #if backward use video_path[-16:][::-1],depth_path[-16:][::-1],mask_path[-16:][::-1],relation_path[-16:][::-1]
        videos = self.read_nomarl_rgbvideo(video_path[-16:][::-1])
        depth_video = self.pross_depth_data( depth_path[-16:][::-1])
        mask_video = self.pross_depth_data(mask_path[-16:][::-1] )
       
        # videos = self.read_nomarl_rgbvideo(video_path[0:16])
        # depth_video = self.pross_depth_data(depth_path[0:16])
        # mask_video = self.pross_depth_data(mask_path[0:16])
        # videos = self.read_nomarl_rgbvideo(selected_video_path)
        # depth_video = self.pross_depth_data(selected_depth_path)
        # mask_video = self.pross_depth_data(selected_mask_path)
        object_number_list, relation_number_list, relation_texts, object_texts, box_list, angle_list, gt_labels = self.process_relation(
            relation_path[-16:][::-1])
        # videos = self.read_nomarl_rgbvideo(video_path[-16:])
        # start_index=os.path.basename(video_path[-16:][0]).split(".")[0]
        # end_index = os.path.basename(video_path[-16:][-1]).split(".")[0]
        idx=(start_index,end_index)
        # depth_video = self.pross_depth_data(depth_path[-16:])
        # mask_video = self.pross_depth_data(mask_path[-16:])
        example = {
        "rgb_video":videos / 127.5- 1.0,
        "depth_video":depth_video / 127.5- 1.0,
        "mask_video":mask_video / 255,
        "idx": idx,
        # "depth_video": depth_video,
        # "mask_video": mask_video,
        "object_id": object_number_list,
        "object_text": object_texts,
        "relation_number": relation_number_list,
        "relation_text": relation_texts,
        "boxes":box_list,
        "angles":angle_list,
        "prompt":c_text,
        "a_prompt":a_text,
        "aa_text":aa_text,
        "lava_c_text": lava_c_text,
        "lava_a_text": lava_p_text,
        "gt_labels":gt_labels,
        "save_name":save_name
        }
        return example




if __name__=="__main__":

    def collate_fn_list(batch):
        # coords=[item for item in batch]
        return batch
    train_dataset = DADA2KS_Graph_Train(root_path=r"./Train_relation_json", interval=1,
                                  phase="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=False,collate_fn=collate_fn_list,
        pin_memory=True, drop_last=True,num_workers=12)

    for id, batch in enumerate(train_dataloader):
        print(id)

        print(batch[0]["rgb_video"].shape)

        print(batch[0]["a_prompt"])

        print(batch[0]["prompt"])
