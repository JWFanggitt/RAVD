# RAVD

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#environment-setup)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](#environment-setup)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ADVersa-yellow.svg)](https://huggingface.co/wwwadad/ADVersa)

</div>

> This repository contains the PyTorch implementation of **RAVD**, including graph-aware video diffusion, Video-LLaVA-based reasoning, Stage-3 training code, inference code, and released checkpoints used in the ADVersa project.

Note:
> Training code for other stages can be reproduced by the Stage-3 pipeline, adjusting the training settings, parameter updates, and trainable network weights according to the paper.
## Overview

The repository currently includes:

- `run_inference_demo.py`: inference entry.
- `train_with_graph.py`: training entry.
- `pipeline.py`: text-to-video diffusion pipeline.
- `grapth_my_dataset.py`: dataset loading and path definitions.
- `configs/lora_training_config.yaml`: default config used by both training and inference scripts.

This README is a general project guide. A separate section below briefly documents the **Key Action** and **Prevention** tasks only. For other MMAU tasks and benchmark details, please visit the [homepage](www.lotvsmmau.net).

## Environment Setup

The codebase is written for Python 3.10+ and PyTorch 2.x.

```bash
conda create -n ravd python=3.10 -y
conda activate ravd

pip install -r requirements.txt
pip install peft bitsandbytes lightning imageio ftfy regex
pip install -e ./CLIP-main

# Optional but recommended for memory/performance
pip install xformers
pip install flash_attn
```

## Pretrained Assets

Released assets are available from: <https://huggingface.co/wwwadad/ADVersa>

Place the required files in the following locations:

| Asset | Required local path | What the code expects |
|---|---|---|
| Base diffusion checkpoint | [`./checkpoint/`](https://huggingface.co/wwwadad/ADVersa/tree/main/RAVD) | Must contain `scheduler/`, `tokenizer/`, `text_encoder/`, `unet/`, and `vae/` |
| Graph model weights | [`./graph_model.bin`](https://huggingface.co/wwwadad/ADVersa/tree/main/RACD) | Loaded by `SceneVAEModel` |
| ControlNet weights | [`./contro/`](https://huggingface.co/wwwadad/ADVersa/tree/main/RAVD/contronet) | Must contain both `config.json` and `controlnet.pth` |
| Video-LLaVA base model | `./Video_LLaVA/LanguageBind/Video-LLaVA-7B-hf/` | Loaded locally by `VideoLlavaForConditionalGeneration.from_pretrained(...)` |
| Video-LLaVA LoRA adapter | [`./lora/`](https://huggingface.co/wwwadad/ADVersa/tree/main/LORA) | Current code loads the PEFT adapter from this folder |

Important:

- The previous README mentioned `./lora_Video_LLaVA/`, but the current code actually loads the adapter from `./lora/`.
- `./contro/` is a directory, not a single file. The loader expects `./contro/config.json` and `./contro/controlnet.pth`.

## Data Layout

The current code mixes relative paths and hardcoded local paths. If you want to run the repository without editing code, keep the following directory layout. Otherwise, update the paths in `grapth_my_dataset.py`.



### Training data

Download our [Train_Data](https://huggingface.co/wwwadad/ADVersa/blob/main/Train_DATA.zip)

The training script creates the dataset with:

```python
DADA2KS_Graph_Train(root_path="./Train_relation_json", phase="train")
```

The current training loader expects:

```text
Train_relation_json/
├── train_v_lava.txt
└── <sample_id>/
    ├── *.json
    └── ...

Graph_Train/
├── Train_Videos/
│   └── <sample_id>/
│       └── images/
│           ├── *.jpg
│           └── ...
├── Train_Videos_Depth/
│   └── <sample_id>/
│       ├── *.jpg
│       └── ...
└── Train_Seg_Track_Mask_New/
    └── <sample_id>/
        ├── *.png
        └── ...
```

If your downloaded release uses `Train_Relation/` instead of `Train_relation_json/`, rename it or create a symlink so the current script can find it.

## Annotation Files

Two text index files are important in the current code:

- `train_v_lava.txt`
- `test_lava.txt`

Based on the current parser in `grapth_my_dataset.py`, `train_v_lava.txt` is read as:

```text
video_id // reason_text // prevention_text // accident_type_text // relation_reason_text
```

and `test_lava.txt` is read as:

```text
video_id // reason_text // accident_type_text // auxiliary_text
```

Per-frame relation JSON files are expected to provide at least:

- `objects`
- `coordinates`
- `relations`

These fields are used to build object ids, relation ids, normalized boxes, angle features, and text prompts for graph-conditioned reasoning.


### Inference data
Download our [Test_Data](https://huggingface.co/wwwadad/ADVersa/blob/main/Test_DATA.zip)

The inference script creates the dataset with:

```python
DADA2KS_Graph_Inference(root_path="./Test_Data/Test_Relation", phase="val")
```
For the current code, the expected structure is:

```text
Test_Data/
├── Test_Relation/
│   ├── test_lava.txt
│   └── <sample_id>/
│       ├── *.json
│       └── ...
├── Test_Video/
│   └── <sample_id>/
│       └── images/
│           ├── *.jpg
│           └── ...
├── Test_Depth/
│   └── <sample_id>/
│       ├── *.jpg
│       └── ...
└── Test_Mask/
    └── <sample_id>/
        ├── *.png
        └── ...
```
Note

We also provide some test samples for qualitative evaluation.  
After extracting the 
```python Test_DATA.zip```, users can access the prediction and recovery videos produced by **RAVD**, **Cosmos**, and **Seer**.

## Quick Start

### Inference

Run inference with:

```bash
python run_inference_demo.py --config ./configs/lora_training_config.yaml
```

The main outputs are written to `./output/`:

- `./output/<sample_id>/frame_*.jpg`
- `./output/<sample_id>/comparison.gif`
- `./output/LLM_results.json`

### Training

Run training with:

Download the [pretraining weights](https://huggingface.co/wwwadad/ADVersa/tree/main/Pretrain)


```bash
python train_with_graph.py --config ./configs/lora_training_config.yaml
```

## Task Notes

This section only adds a brief note for the **Key Action** and **Prevention** tasks. For other task definitions and full benchmark details, please visit our [homepage](www.lotvsmmau.net).

### Key Action

**Task definition.** Given an accident video clip, the model first reasons about the accident cause and then extracts the key action word or phrase most responsible for that cause, such as `drives too fast` or `out of control`.

**Data form.** In practice, this task is organized around video-cause-key-action style supervision. A typical tuple can be viewed as:

```text
(video, reason_text, key_action_text)
```

**Example.**

```json
{
  "id": "video_000000",
  "video": "../videos/43/000030.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "<video>\nWhat is the reason for this retrospective accident video?"
    },
    {
      "from": "gpt",
      "value": "vehicles drive too fast with short braking distance"
    },
    {
      "from": "human",
      "value": "<video>\nWhat are the keys to reasons?"
    },
    {
      "from": "gpt",
      "value": "drive too fast"
    }
  ]
}
```

### Prevention

**Task definition.** Given an accident clip and its visible cause, the model predicts a short prevention-oriented response describing what action or precaution could help avoid the accident.

**Data form.** In the current repository, the prevention signal is stored in `train_v_lava.txt` as the third field:

```text
video_id // reason_text // prevention_text // accident_type_text // relation_reason_text
```

So the prevention task can be viewed as:

```text
(video, reason_text, prevention_text)
```

Download the key_action and prevention tasks [data](https://drive.google.com/file/d/1cNxH-sCtCYHqPAogq6YhH2WJOvrH8XjC/view?usp=sharing)


## Model Zoo

The following checkpoints are organized into two groups.

### Prediction and Retrospection

| Model | Prediction Checkpoint | Retrospection Checkpoint | Notes |
|---|---|---|---|
| Seer | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/OUT) | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/OUT_H) | Released checkpoints for prediction and retrospection |
| Cosmos | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/cosmos_predict_v2p5_P) | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/cosmos_predict_v2p5) | Released checkpoints for prediction and retrospection |
| MCVD | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/MCVD_W) | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/MCVD_W) | Released checkpoints for prediction and retrospection |
| FAR | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/FAR/save_path/P_models) | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/FAR/save_path/R_models) | Released checkpoints for prediction and retrospection |

### Key Action and Prevention

| Task | Model | Checkpoint | Notes |
|---|---|---|---|
| Key Action | AVD2 | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/OUT_Key) | Checkpoint and results for key-action grounding |
| Prevention | AVD2 | [CKPT](https://huggingface.co/wwwadad/ADVersa/tree/main/Out_Q) | Add the released prevention checkpoint here |

Other Tasks model weights

Coming soon.

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{li2026adversa,
  title={ADVersa: Abductive Driving Accident Video Understanding},
  author={Li, Lei-Lei and Fang, Jianwu and Xiao, Junbin and Yu, Hongkai and Lv, Chen and Xue, Jianru and Li, Zhengguo and Chua, Tat-Seng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE}
}
```

Paper link: <https://ieeexplore.ieee.org/abstract/document/11391656/>

## Acknowledgement

This repository builds upon or includes components related to:

- Diffusers
- CLIP
- Video-LLaVA
- ModelScope
- Tune-A-Video
