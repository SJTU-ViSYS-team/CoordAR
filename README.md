<div align="center">

# CoordAR

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](https://img.shields.io/badge/paper-arxiv.2511.12919-B31B1B.svg)](https://www.arxiv.org/abs/2511.12919)
[![Conference](https://img.shields.io/badge/AAAI-2026-4b44ce.svg)](https://aaai.org/conference/aaai/aaai-26/)

</div>

Official implementation of AAAI2026 Oral paper "CoordAR: One-Reference 6D Pose Estimation of Novel Objects via Autoregressive Coordinate Map Generation"

## 📣 News
- **2026-02-23** We release our code.
- **2026-01-29** We plan to realse our code in February.

## 📋TODOs for code release
- [x] Dataset Interface
- [x] Model code
- [ ] Trained weights

## Overview


CoordAR is a novel autoregressive framework for single-reference 6D pose estimation of unseen objects. CoordAR requires only a
single reference RGB-D  image instead of a full 3D model. Our method formulates 3D-3D correspondences between
the reference and query views as a token
map, which is decoded autoregressively in a probabilistic
manner. 

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/SJTU-ViSYS-team/CoordAR
cd CoordAR
git submodule update --init --recursive

# [OPTIONAL] create conda environment
conda create -n coordar python=3.9
conda activate coordar

# install pytorch according to instructions
# https://pytorch.org/get-started/
pip install torch

# install requirements
pip install -r requirements.txt

```

## Data Preparation
### FoundationPose GSO & FoundationPose Objaverse & BOP datasets
Download BOP dataset files from [BOP website](http://bop.felk.cvut.cz/home/).
Download FoundationPose dataset files from [FoundationPose \[Google Drive\]](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-). Prepare datasets folder like this:
```
data/
├── BOP
    ├──lm
        └──test # download from BOP website
```


## Download Checkpoints
Download our trained models from [\[Baidu Wangpan\]](https://pan.baidu.com/s/1wSgoCB5VIhtC2jDLYlVWrg), code: `1sks`, put it into [./logs](). Prepare weight folder like this:
```
logs/
├── checkpoints
    ├──coordar
        ├──model.pth
    ├──coordar
        └──last.ckpt
```
## Reproduce the results
```bash
# run predict with our trained model
export `<.env.example`
python src/predict.py experiment=coordar/ar_paper logger=csv
# or with model trained by yourself
python src/predict.py experiment=coordar/ar_paper logger=csv ckpt_path=path_to_ckpt
```

You can override any parameter from command line like this to switch between multiple datasets.

```bash
python src/predict.py data.data_predict=bop_lm_predict
```


## Training
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=coordar/ar_paper logger=tensorboard
```
