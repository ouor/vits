# How to use
(Suggestion) Python == 3.7

## Clone this repository

```sh
git clone https://github.com/ouor/vits.git
```

## Choose cleaners

- Fill "text_cleaners" in config.json
- Edit text/symbols.py
- Text cleaner is korean by default
- Remove unnecessary imports from text/cleaners.py

## Create conda environment

```sh
conda create -n vits python=3.7
conda activate vits
conda install conda-forge::uv
```

## Install PyTorch

```sh
uv pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
```

## Install requirements

```sh
uv pip install -r requirements.txt
```

## Create datasets

### Single speaker
"n_speakers" should be 0 in config.json
```
path/to/XXX.wav|transcript
```
- Example
```
trains/korean/datasets/train/001.wav|안녕하세요.
...
trains/korean/datasets/val/001.wav|안녕하세요.
```
### Mutiple speakers
Speaker id should start from 0 
```
path/to/XXX.wav|speaker id|transcript
```
- Example
```
trains/korean/datasets/train/001.wav|0|안녕하세요.
...
trains/korean/datasets/val/001.wav|0|안녕하세요.
```
## Preprocess

If you have done this, set "cleaned_text" to true in config.json
```sh
# Single speaker
python preprocess.py --text_index 1 --filelists trains/korean/datasets/train/filelist_train.txt trains/korean/datasets/val/filelist_val.txt

# Mutiple speakers
python preprocess.py --text_index 2 --filelists trains/korean/datasets/train/filelist_train.txt trains/korean/datasets/val/filelist_val.txt
```
## Build monotonic alignment search

```sh
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```
## Train

```sh
# Single speaker
python train.py -c trains/korean/config.json -m trains/korean/models

# Mutiple speakers
python train_ms.py -c trains/korean/config.json -m trains/korean/models
```

## Tensorboard

```sh
tensorboard --logdir=trains/korean/models
```

## Inference

See [inference.ipynb](inference.ipynb)

# Running in Docker

```sh
docker run -itd --gpus all --name "Container name" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all "Image name"
```