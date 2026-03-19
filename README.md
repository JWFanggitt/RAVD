# RAVD

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](#installation)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ADVersa-yellow.svg)](https://huggingface.co/wwwadad/ADVersa)

</div>

> This repository provides the official PyTorch implementation of **RAVD**, including training and inference pipelines, as well as pretrained checkpoints.
- ✅ **Stage-3 Training Code**: We provide training scripts for the Stage-3 model.
- 🔄 **Stage-1 & Stage-2 Adaptation**: The implementations for Stage-1 and Stage-2 can be derived by modifying the Stage-3 training pipeline according to the configurations and model designs described in the paper (e.g., architecture settings, loss functions, and training schedules).
- 🚀 **Inference & Checkpoints**: Pretrained models and inference scripts are included for easy evaluation and reproduction.
---

## Data Organization

The training and testing data are expected to follow a structure similar to the following.

### Training data
Download [Train_DATA]([https://your-link.com](https://huggingface.co/wwwadad/ADVersa/blob/main/Train_DATA.zip))
```text
Train_DATA/
├── Train_Relation/
├── Train_Seg_Track_Mask_New/
├── Train_Videos/
├── Train_Videos_Depth/
└── Train_Seg_Track/
```
Download [Test_DATA](https://huggingface.co/wwwadad/ADVersa/blob/main/Test_DATA.zip)
### Testing data

```text
Test_Data/
├── Test_Depth/
├── Test_Mask/
├── Test_Relation/
├── Test_Video/
├── ADVersa/ # Predicted and Recovery samples by RAVD.
├── Test_Cosmos/ #Predicted and Recovery samples by Cosmos-Predict-2B.
├── Test_Seer/ #Predicted and Recovery samples by Seer.
```

```
Text files, such as `train_v_lava.txt` and `test_lava.txt`, are used for training and testing in `Train_Relation` and `Test_Relation`, respectively.
---


## Inference 

The main inference entry is:

```bash
python run_inference_demo.py
```

The inference pipeline expects several local assets to be available, including:
- [`./checkpoint`](https://huggingface.co/wwwadad/ADVersa/tree/main/RAVD)    
- [`./graph_model.bin`](https://huggingface.co/wwwadad/ADVersa/blob/main/RAVD/Graph)
- [`./lora_Video_LLaVA/`](https://huggingface.co/wwwadad/ADVersa/tree/main/LORA)
- [`./contro/`](https://huggingface.co/wwwadad/ADVersa/blob/main/RAVD/controlnet.pth,https://huggingface.co/wwwadad/ADVersa/blob/main/RAVD/controlnet.pth)
- `./Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf`
---




## Citation

If you use this repository in your research, please cite your paper here:

📄 **Paper**: [ADVersa: <paper title>](https://ieeexplore.ieee.org/abstract/document/11391656/)

```

---
## Acknowledgement
This repository builds upon or includes components related to:
- Diffusers
- CLIP
- Video-LLaVA
- ModelScope
- Tune-A-Video
Thanks to the open-source community for making these resources available.
# RAVD
