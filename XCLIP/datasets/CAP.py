import os
import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange, repeat, reduce
import glob
from PIL import Image

class DADA2KS(Dataset):
    def __init__(self, root_path, interval,phase,
                  data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase=phase
        self.data_list, self.end, self.NC_text, self. R_text ,self.P_text,self.C_text= self.get_data_list()



    def get_data_list(self):
        if self.phase =="train":
            list_file = os.path.join(self.root_path+"/"+'OOD_new.txt')
        # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs,end,NC_text,R_text,P_text,C_text= [],[],[],[],[],[]

            with open(list_file, 'r',encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    parts = line.strip().split('，')
                    if len(parts) == 2:
                        ID, end_number = parts[0].split(' ')
                        fileIDs.append(ID)
                        end.append(end_number)
                        subparts = parts[1].split('//')
                        if len(subparts) == 4:
                            NC_text.append(subparts[0])
                            R_text.append(subparts[1])
                            P_text.append(subparts[2])
                            C_text.append(subparts[3])
            return fileIDs,end,NC_text,R_text,P_text,C_text

        if self.phase == "val":
            list_file = os.path.join(self.root_path + "/" + 'Tc.txt')
            # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, labels, clips, toas, texts = [], [], [], [], []
            # samples_visited, visit_rows = [], []
            with open(list_file, 'r', encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split(',')  # e.g.: 1/002 1 0 149 136
                    sample1 = sample[0].strip().split(' ')
                    for i in range(1,len(sample)):
                        sample[i]= sample[i].replace('\xa0', ' ')
                    textss=sample[1:len(sample)]
                    texts.append(textss)
                    # word = sample[1:-1]
                    # word.strip()
                    fileIDs.append(sample1[0])  # 1/002
                    labels.append(int(sample1[1]))  # 1: positive, 0: negative
                    clips.append([int(sample1[2]), int(sample1[3])])  # [start frame, end frame]
                    toas.append(int(sample1[4]))  # time-of-accident (toa)
                    # texts.append(word.strip())

            return fileIDs, labels, clips, toas, texts



    def __len__(self):
        return len(self.data_list)

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

    def read_rgbvideo(self, video_file,end):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        nv=video_file[0:16]
        rv=video_file[end-32:end-16]
        av=video_file[end-16:end]
        nv=self.pross_video_data(nv)
        rv=self.pross_video_data(rv)
        av=self.pross_video_data(av)
        return nv,rv,av


    def gather_info(self, index):
        # # accident_id = int(self.data_list[index].split('/')[0])
        # accident_id =self.data_list[index]
        end=int(self.end[index])
        NC_text= self.NC_text[index]
        R_text= self.R_text[index]
        P_text= self.P_text[index]
        C_text=self.C_text[index]
        return end,NC_text,R_text, P_text, C_text


    def __getitem__(self, index):

        end, NC_text, R_text, P_text, C_text=self.gather_info(index)

        video_path = os.path.join(self.root_path,self.data_list[index]+"/"+"images")
        video_path=glob.glob(video_path+'/'+"*.jpg")
        video_path= sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))

        nv,rv,av=self.read_rgbvideo(video_path,end)
        example = {
        "nv": nv / 127.5 - 1.0,
        "rv": rv / 127.5 - 1.0,
        "av": av / 127.5 - 1.0,
        "N_t":NC_text,
        "R_t":R_text,
        "P_t":P_text,
        "C_t":C_text
    }
        return example



if __name__=="__main__":
    train_dataset = DADA2KS(root_path=r"/media/ubuntu/My Passport/CAPDATA", interval=1,phase="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        pin_memory=True, drop_last=True)

    for id, batch in enumerate(train_dataloader):
            # print(step)
            print(batch["nv"].shape)
            print(batch["rv"].shape)
            print(batch["av"].shape)
            print(batch["N_t"])
            print(batch["R_t"])
            print(batch["P_t"])
            print(batch["C_t"])
