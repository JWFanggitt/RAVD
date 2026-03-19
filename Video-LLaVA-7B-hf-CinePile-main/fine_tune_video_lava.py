#!/usr/bin/env python
# coding: utf-8

# ## Fine-tune Video-LLaVa on CinePile dataset
#
# In this notebook, we are going to fine-tune the [Video-LLaVa](https://huggingface.co/docs/transformers/main/en/model_doc/video_llava) model on [CinePile](https://huggingface.co/datasets/tomg-group-umd/cinepile) dataset which is a question-answering-based, long-form video understanding dataset.
#
# Video-LLaVa is an open-source multimodal model that can accept both, images and videos as input in an interleaved manner. The model architecture is pretty much similar to [LLaVa](https://huggingface.co/docs/transformers/main/en/model_doc/llava).
#
# The goal for the model in this notebook is to answer given multiple choice questions basedd on the video. The questions can be realetd to temporal aspects, character and relationship dynamics, narrative and plot analysis or theme exploration.
#
# Sources:
#
# * Video-LLaVa [documentation](https://huggingface.co/docs/transformers/main/en/model_doc/video_llava)
# * Video-LLaVa [checkpoint on the hub that we use for fine-tuning](https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf)
#
# **Note: this notebook is a direct adaptation of Niels' [LLaVa notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb).**

# # Pre-requisites
#
# This notebook assumes that you have downloaded the videos pointed in the CinePile dataset and those are accessible in a local folder.
# We used [Video2Dataset](https://github.com/iejMac/video2dataset) for this. Our YAML config file (video2dataset-config.yaml) and links (dataset.csv) can be found in this repo.

# ## Define variables
#
# We'll first set some variables useful througout this notebook and do all the necessary imports.

# In[ ]:
from datasets import Dataset, DatasetDict
import random
import torch
from torch.utils.data import Dataset, DataLoader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import json
import av
import re
import bisect
import numpy as np
import wandb
import datetime
import cv2

from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration, VideoLlavaProcessor,AutoConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model,PeftModel

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, load_from_disk

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.profilers import SimpleProfiler


NUM_FRAMES_VIDEO = 8
MAX_LENGTH_PROCESSOR=2048

MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"

#Path to the download folder of Video2Dataset
VIDEO_SNAPSHOT_PATH = "/path/to/cinepile/fulldatasetvideoscenes/"

#Base path for temporary files and model snapshots
LOCAL_PATH = r"/media/lotvs/WD_4T/Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf"


# # Auxiliar video processing functions

# In[ ]:
class DummyDataset(Dataset):
    def __init__(self, num_samples=20, num_frames=NUM_FRAMES_VIDEO):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.choices = ["A", "B", "C", "D", "E"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机生成一个“问题”和一个“正确答案”
        question = f"What is happening in dummy video #{idx}?"
        answer = random.choice(self.choices)

        # 随机生成一段“视频帧” (num_frames, 3, 224, 224)
        # VideoLLaVa 的 processor 接受 numpy array
        clip = torch.randn(self.num_frames, 3, 224, 224).numpy()

        # 构造 prompt，和 VideoLlavaDataset 一致的格式（去掉 subtitle）
        prompt = (
            "USER: <video>Why is this video funny? ASSISTANT:Because the cat falls unexpectedly."
        )

        return prompt, clip

def resize_and_crop(img, target_height=224):
    img = img.to_ndarray(format="rgb24")
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Input image is not a valid NumPy array.")

    # Ensure the image is in uint8 format
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    height, width = img.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions are invalid: {height}x{width}.")

    # Calculate the new width while maintaining aspect ratio
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)

    # Resize image
    try:
        resized_img = cv2.resize(img, (new_width, target_height))
    except cv2.error as e:
        raise RuntimeError(f"Error resizing image: {e}")

    # Crop to make width same as height
    if new_width < target_height:
        raise ValueError(f"Resized width {new_width} is smaller than target height {target_height}.")

    start_x = (new_width - target_height) // 2
    cropped_img = resized_img[:, start_x:start_x + target_height]

    return cropped_img

def read_equidistant_frames_pyav(video_path, num_frames):
    """Reads a video for given start-end timestamps interval and uniformly samples num+frames of it"""
    container = av.open(video_path)
    video = container.streams.get(0)[0]

    av_timestamps = [
        int(packet.pts * video.time_base) for packet in container.demux(video) if packet.pts is not None
    ]

    av_timestamps.sort()
    start_id = bisect.bisect_left(av_timestamps, 1)
    end_id = bisect.bisect_left(av_timestamps, 1e10)

    # in case it is a very short video, lets take a longer duration and sample
    if end_id  - start_id < 10:
        end_id += 10
        start_id -= 10

    end_id = min(len(av_timestamps) - 1, end_id)
    start_id = max(1, start_id)
    indices = np.linspace(start_id, end_id, num_frames).astype(int)

    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_id:
            break
        if i >= start_id and i in indices:
            frames.append(resize_and_crop(frame))
    assert len(frames) == num_frames, f"Got {len(frames)} frames but should be {num_frames}. Check the indices: {indices};, start_id: {start_id}, end_id: {end_id}. Len of video is {len(av_timestamps)} frames."
    return np.stack(frames)


def read_specific_frames_pyav(video_path, frames):
    frames_out = []
    container = av.open(video_path)
    video = container.streams.get(0)[0]
    container.seek(0)

    for idx, frame in enumerate(container.decode(video=0)):
        if idx in frames:
            frames_out.append(resize_and_crop(frame))

    assert len(frames) == len(frames_out), f"{video_path}: Got {len(frames_out)} frames extracted but should be {len(frames)}"
    return np.stack(frames_out)

def get_frames_for_video(path, cuts, num_frames):
    num_segments = len(cuts)
    if num_segments < num_frames:
        return read_equidistant_frames_pyav(path, num_frames)
    else:
        step = num_segments / num_frames
        frame_starts = []
        for i in range(num_frames):
            index = int(i * step)
            frame_starts.append(cuts[index][0])

        return read_specific_frames_pyav(path, frame_starts)

def collate_read_video(example, lookup, num_frames):
    # Some datasets have a start-end interval, so we try to get it if exists. Otherwise just set a very large end timestamp
    if example['yt_clip_link'] not in lookup:
        example['clip'] = None
        return example

    if lookup[example['yt_clip_link']]['precalculation'] is None:
        clip = get_frames_for_video(lookup[example['yt_clip_link']]['path'],
                        lookup[example['yt_clip_link']]['cuts'],
                        num_frames)
        lookup[example['yt_clip_link']]['precalculation'] = clip

    example['clip'] = lookup[example['yt_clip_link']]['precalculation']
    return example

def filter_no_video(example, lookup):
    if example['yt_clip_link'] not in lookup:
        return False
    return True


# # Dataset preparation
# In this section, we combine the metadata from CinePile with the frames extracted from the videos downloaded using Video2Dataset.

# In[ ]:

def generate_video_md_lookup_table(video_base_dir):
    """Assuming dataset stored in multiple subfolders (shards from video2dataset). We add to the lookup table only the cases where we have metadata and video"""
    lookup_table = {}

    # Traverse the directory recursively
    for root, _, files in os.walk(video_base_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                video_path = json_path[:-5] + '.mp4'  # Assuming json_path always ends with .json

                # Check if the corresponding .mp4 file exists
                if os.path.exists(video_path):
                    # Read the JSON file
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    # Extract the cuts section from the JSON
                    cuts = data['cuts']['cuts_original_fps']

                    # Extract yt_clip_link from the JSON (assuming it's a unique identifier)
                    yt_clip_link = data['url']

                    # Store the result in the lookup table
                    lookup_table[yt_clip_link] = {"path": video_path, "cuts": cuts, "precalculation":None}

    return lookup_table

# In[ ]:


# Load each config and save in a mapping
# ds = load_dataset("tomg-group-umd/cinepile")
# lookup = generate_video_md_lookup_table(VIDEO_SNAPSHOT_PATH)
# num_processes = 4
# train_ds = ds['train']

# print(f"Initial train dataset length: {len(train_ds)}")
# dataset = train_ds.filter(lambda example: filter_no_video(example, lookup))
# print(f"Dataset size after filtering non-video cases: {len(dataset)}")

#Sharding the mapping work
# num_blocks = 200
# # Calculate block size
# block_size = len(dataset) // num_blocks
# print(block_size)
# remainder = len(dataset) % num_blocks
# # Iterate through each block
# for i in range(num_blocks):
#     start_idx = i * block_size
#     end_idx = start_idx + block_size
#
#     # For the last block, include any remaining samples
#     if i == num_blocks - 1:
#         # end_idx = len(dataset)

    # # Select the current shard
    # print(f"Selecting between {start_idx}-{end_idx}")
    # save_directory = LOCAL_PATH + f'/saved_datasets/cinepile/train_dataset_small/shard_{i}/'
    #
    # # Useful in case of a crash while preparing the dataset
    # # if os.path.exists(save_directory):
    # #     print("\t Folder for that shard found. Skipping recalculation")
    # #     continue
    # #
    # # curr_shard = dataset.select(range(start_idx, end_idx))
    # curr_shard = curr_shard.map(
    #     collate_read_video,
    #     batched=False,
    #     fn_kwargs={"lookup": lookup, "num_frames": NUM_FRAMES_VIDEO},
    #     num_proc=num_processes,
    #     writer_batch_size=10
    # )
    #
    #
    # curr_shard.save_to_disk(save_directory)
    # print(f"Shard {i} - mapping complete")
    # dataset.cleanup_cache_files()


# In[ ]:


# Reload stored partitions
# num_blocks = 200
# sharded_dataset = []
# for i in range(num_blocks):
#     storage_directory = LOCAL_PATH + f'/saved_datasets/cinepile/train_dataset_small/shard_{i}/'
#     sharded_dataset.append(load_from_disk(storage_directory))
# dataset = concatenate_datasets(sharded_dataset)
# print(f"load complete - len:{len(dataset)}")


# In[ ]:

processor = VideoLlavaProcessor.from_pretrained(r"/media/lotvs/WD_4T/Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf")
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right

# ## Custom Dataset Class
#
# In the next step, we'll define a custom dataset class and the necessary functions to prepare our data for fine-tuning the Video-LLaVA model. The VideoLlavaDataset class extends the [PyTorch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) class to facilitate loading and processing "MMBench". This class will handle the conversion of dataset samples into the format required for training and evaluation by preparing a prompt and making array from videos.
#
# NOTE: Video-LLaVa accepts videos in one of the following formats:
# - an array or tensor of shape: (batch-size, frames, channel, height, width) where batch-size is an optional dimension
# - a list of arrays of shape: (frames, channel, height, width)
# - a nested list of video frames, where each frame is an image
#
#
# Next, we define collate functions to handle the batching of data during training and evaluation. These functions ensure that the input data is properly formatted and padded.
#
# It's only here that we're going to use the processor to turn the (video, target token sequence) into the format that the model expects (which is pixel_values, input_ids etc.). The reason we do that here is because it allows for dynamic padding of the batches: each batch contains ground truth sequences of varying lengths. By only using the processor here, we will pad the input_ids up to the largest sequence in the batch.
#
# We also decide to limit the length of the text tokens (input_ids) to a max length due to memory constraints, feel free to expand if your target token sequences are longer (I'd recommend plotting the average token length of your dataset to determine the optimal value).
#
# The formatting of the input_ids is super important: we need to respect a so-called [chat template](https://huggingface.co/docs/transformers/main/en/chat_templating). As of now, Video-LLaVa does not yet support chat templates, so we manually write down the prompt in the correct format (which starts with USER and ends with ASSISTANT).You could also omit this and just train the model on (video, instruction) pairs without text prompt.
#
# Labels are created for the model by simply copying the inputs to the LLM (input_ids), but with padding tokens replaced by the ignore index of the loss function. This ensures that the model doesn't need to learn to predict padding tokens (used to batch examples together).
#
# Why are the labels a copy of the model inputs, you may ask? The model will internally shift the labels one position to the right so that the model will learn to predict the next token. This can be seen here.
#
# The collate function for evaluation is different, since there we only need to feed the prompt to the model, as we'll use the `generate()` method to autoregressively generate a completion.

# In[ ]:

# In[ ]:

def train_collate_fn(examples):
    texts, videos = list(zip(*examples))
    batch = processor(text=texts, videos=videos,padding=True,truncation=True,return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # assistant_token = processor.tokenizer("ASSISTANT:").input_ids[1:-1]  # 去除 BOS/EOS
    # for j in range(len(input_id) - len(assistant_token)):
    #     if torch.equal(input_id[j:j + len(assistant_token)], torch.tensor(assistant_token, device=input_id.device)):
    #         labels[i, :j + len(assistant_token)] = -100
    #         break
    #
    #
    # labels[labels == processor.tokenizer.pad_token_id] = -100
    # batch["labels"] = labels
    #










    batch["labels"] = labels
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]
    labels = batch["labels"]
    # 手动添加 video_token_len 字段

    return input_ids, attention_mask, pixel_values_videos, labels


def eval_collate_fn(examples):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    print(f"{current_time}-inside eval")

    # We only feed the prompt to the model
    textsOriginal, videos = list(zip(*examples))
    texts = [text[:-2] for text in textsOriginal]
    batch = processor(text=texts, videos=videos, padding=True, truncation = True, max_length=MAX_LENGTH_PROCESSOR, return_tensors="pt")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]
    answer_choice = [text[-1] for text in textsOriginal]
    return input_ids, attention_mask, pixel_values_videos, answer_choice


# ## Load model
# Next, we're going to load the Video-LLaVa model from the hub. This is a model with about 7 billion trainable parameters (as it combines a LLaMa-7B language model with a relatively low-parameter vision encoder). Do note that we load a model here which already has undergone supervised fine-tuning (SFT) on VideoChat instruction dataset. We can benefit from the fine-tuning that the model already has undergone.
#
# ## Q-LoRa
# As this model has 7 billion trainable parameters, that's going to have quite an impact on the amount of memory used. For reference, fine-tuning a model using the AdamW optimizer (which is often used to optimize neural networks) with mixed precision, you need about 18 times the amount of parameters in GB of GPU RAM. So in this case, we would need 18x7 billion bytes = 126 GB of GPU RAM if we want to update all the parameters of the model!! That's huge right? And for most people infeasible.
#
# Luckily, some clever people came up with the LoRa method (LoRa is short for low-rank adapation). It allows to just freeze the existing weights and only train a couple of adapter layers on top of the base model. Hugging Face offers the separate [PEFT library](https://huggingface.co/docs/peft/main/en/index) for easy use of LoRa, along with other Parameter-Efficient Fine-Tuning methods (that's where the name PEFT comes from).
#
# Moreover, one can not only freeze the existing base model but also quantize it (which means, shrinking down its size). A neural network's parameters are typically saved in either float32 (which means, 32 bits or 4 bytes are used to store each parameter value) or float16 (which means, 16 bits or half a byte - also called half precision). However, with some clever algorithms one can shrink each parameter to just 8 or 4 bits (half a byte!), without significant effect on final performance. Read all about it here: https://huggingface.co/blog/4bit-transformers-bitsandbytes.
#
# Of course, if you have the memory available, feel free to use full fine-tuning or LoRa without quantization! In case of full fine-tuning, the code snippet below instantiates the model with Flash Attention which considerably speeds up computations.
#
# There exist many forms of quantization, here we leverage the [BitsAndBytes integration](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig).

# In[ ]:


## Load model
# QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

config = AutoConfig.from_pretrained(r"/media/lotvs/WD_4T/Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf")

model = VideoLlavaForConditionalGeneration.from_pretrained(
    r"/media/lotvs/WD_4T/Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    config=config,
)

# ## Apply PEFT
# After loading the base model, we're going to add LoRa adapter layers. We're going to only train these adapter layers (the base model is kept frozen).
#
# The difference here with other models are the layers at which we're going to add adapters (in PEFT this is called target_modules). This typically depends a bit on the model.
#
# We defined a function to find all linear layers in the model, excluding any layers related to multimodal projections and vision models. This function will help us identify which layers should have LoRA applied. We're going to add adapters to all linear layers of the model (nn.Linear), except for the ones present in the vision encoder and multimodal projector. This means that we're mostly going to adapt the language model part of Video-LLaVa for our use case.

# In[ ]:


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


# ## Define PyTorch Lightning Module for Video-LLaVA
# To streamline the training and evaluation of the Video-LLaVA model, we use [LightningModule](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), which abstracts away much of the boilerplate code and provides a structured framework for model training. In this section, we define the VideoLlavaModelPLModule, a custom PyTorch Lightning module that encapsulates the model, training loop, validation loop, and optimizer configuration.
#
# ### VideoLlavaModelPLModule Class
#
# The VideoLlavaModelPLModule class inherits from LightningModule and includes methods for training, validation, and optimizer configuration. This setup ensures a clean and efficient training process.
#
# Basically, PyTorch Lightning will take care of all device placements (.to(device)) for us, as well as the backward pass, putting the model in training mode, etc.
#
# Notice the difference between a training step and an evaluation step:
#
# - a training step only consists of a forward pass, in which we compute the cross-entropy loss between the model's next token predictions and the ground truth (in parallel for all tokens, this technique is known as "teacher forcing"). The backward pass is handled by PyTorch Lightning.
# - an evaluation step consists of making the model autoregressively complete the prompt using the generate() method. After that, we compute an evaluation metric between the predicted sequences and the ground truth ones. This allows us to see how the model is improving over the course of training. The metric we use here is accuracy of answering the question.
#
# Besides that, we define the optimizer to use (AdamW is a good default choice) and the data loaders, which use the collate functions defined above to batch together items of the PyTorch datasets. Do note that AdamW is a pretty heavy optimizer in terms of memory requirements, but as we're training with QLoRa we only need to store optimizer states for the adapter layers. For full fine-tuning, one could take a look at more memory friendly optimizers such as 8-bit Adam.

# In[ ]:


config = {"max_epochs": 5,
          "val_check_interval": 0.2, # how often we want to validate during an epoch,
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-3,
          "batch_size": 1,
          "num_nodes": 1,
          "warmup_steps": 50,
}

# In[ ]:


class VideoLlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values_videos, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        with torch.no_grad():
            MAX_NEW_TOKENS = 256
            input_ids, attention_mask, pixel_values_videos, answers = batch

            # autoregressively generate token IDs
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
            # turn them back into text, chopping of the prompt
            predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

            correct = 0
            for pred, answer in zip(predictions, answers):
                correct += (pred.strip().lower() == answer.lower())

            self.log("val_accuracy", float(correct) / len(answers))


            return correct

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=3)

    def val_dataloader(self):
        return DataLoader(eval_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=3)


# Let's instantiate it (based on a config dictionary which defines all hyperparameters for training).
#
# The batch size was determined based on the compute available.
#
# Do note that one can play around with the hyperparameters, I just use good defaults here: 10 epochs, a learning rate of 1e-4 which I found in the original Idefics2 notebook (linked at the top of this notebook), use mixed precision for training (more memory friendly). One could extend this with things like gradient accumulation and gradient checkpointing.
#
# I recommend [this guide](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one) which goes over all tips and tricks regarding maximizing fine-tuning performance on consumer hardware.

# In[ ]:


model_module = VideoLlavaModelPLModule(config, processor, model)
early_stop_callback = EarlyStopping(monitor="val_accuracy", patience=3, verbose=False, mode="min")


# ## Define callbacks
# Optionally, Lightning allows to define so-called [callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html), which are arbitrary pieces of code that can be executed during training.
# We will use them to store checkpoints of the model

# In[ ]:


from datetime import datetime

class SaveModelCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"{LOCAL_PATH}/weights/{current_time}-checkpoint-{trainer.current_epoch}"
        pl_module.model.save_pretrained(output_dir)
        print(f"Model checkpoint saved at epoch {trainer.current_epoch} to {output_dir}")
    def on_train_end(self, trainer, pl_module):
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"{LOCAL_PATH}/weights/{current_time}-checkpoint-final"
        pl_module.model.save_pretrained(output_dir)
        print(f"Model checkpoint saved at the end of the training to {output_dir}")

# —— 2. 模型与优化器 ——————————————————————————————
device=torch.device("cuda",0)
model=model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
max_epochs = config["max_epochs"]
train_dataset = DummyDataset(num_samples=100)
eval_dataset  = DummyDataset(num_samples=100)

# —— 3. 准备 DataLoader ————————————————————————
train_loader = DataLoader(
    train_dataset,
    collate_fn=train_collate_fn,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=2,
)
eval_loader = DataLoader(
    eval_dataset,
    collate_fn=eval_collate_fn,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=2,
)
# —— 3. 手动训练／验证循环 ——————————————————————————
for epoch in range(1, max_epochs + 1):
    # —— 3.1 训练阶段 ——————————————————————
    model.train()
    total_loss = 0.0
    for input_ids, attention_mask, pixel_values_videos, labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values_videos = pixel_values_videos.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # print(next(model.parameters()).device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print("loss:",loss)
        total_loss += loss.item()


        # output_dir = r"/media/lotvs/WD_4T/Video_LLaVA/lora_weight"
        # model.save_pretrained(output_dir)
        # print(f"LoRA adapter weights saved to {output_dir}")
        #
        # model = PeftModel.from_pretrained(model, output_dir)



    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{max_epochs} — Train Loss: {avg_train_loss:.4f}")

    # —— 3.2 验证阶段 ——————————————————————
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for input_ids, attention_mask, pixel_values_videos, answers in eval_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pixel_values_videos = pixel_values_videos.to(device)

            # 生成预测
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values_videos=pixel_values_videos,
                max_new_tokens=256,
                do_sample=False
            )
            preds = processor.batch_decode(
                generated_ids[:, input_ids.size(1):],
                skip_special_tokens=True
            )
            # 累计准确数
            for pred, ans in zip(preds, answers):
                total_correct += (pred.strip().lower() == ans.lower())
                total_samples += 1

    val_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"Epoch {epoch}/{max_epochs} — Val Accuracy: {val_accuracy:.4f}")

# ## Train!
# Alright, we're set to start training!
#
# Do note that this Trainer class supports many more flags! See the [docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer)

# In[ ]:

#
# trainer = L.Trainer(
#         accelerator="gpu",
#         devices=1,
#         max_epochs=config.get("max_epochs"),
#         accumulate_grad_batches=config.get("accumulate_grad_batches"),
#         gradient_clip_val=config.get("gradient_clip_val"),
#         precision="16-mixed",
#         limit_val_batches=5,
#         num_sanity_val_steps=1,
#         callbacks=[early_stop_callback,SaveModelCallback()],
#         val_check_interval=config.get("val_check_interval"),
# #        fast_dev_run=True,
# )


# In[ ]:


# trainer.fit(model_module)
