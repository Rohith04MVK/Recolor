<h1 align="center"> Recolor</h1>
<h3 align="center">Retouch old black and white imges whith Recolor!</h3>
<p align="center"><img src="/images/index.jpg" /></p>

## **TODO**
 - Better image quality

## Installation and Usage

This project uses `pipenv` for dependency management. You need to ensure that you have pipenv installed.

Here are the commands to facilitate using this project.

#### Clone the repo

```sh
git clone https://github.com/Rohith04MVK/Recolor
cd Recolor
```

#### Install dependencies and Open the shell

```sh
# Install dependencies
pipenv sync -d

# Open the venv shell
pipenv shell
```

#### Train the model

```sh
python3 main.py --train-type general \
  --save-path /kaggle/working/output \
  --pretrain y \
  --epochs 25 \
  --use-gpu y
```

#### Inference

```sh
python infer.py --model-path path/to/model \
  --input-img path/to/input/image \
  --output path/to/save/output/image \
```

## Requirements

You will need the following to run the above:

- Torch 1.9.1
- Python 2.8.5, Pillow 8.2.0, numpy 1.20.3, fastai 2.4
- If you want to train (and don't want to wait for 4 months):
  - A decent GPU
  - All the required NVIDIA software to run Torch on a GPU (cuda, etc)

Tested on:

| Spec             |                             |
| ---------------- | --------------------------- |
| Operating System | Ubuntu 20.04.3              |
| GPU              | NVIDIA Tesla P100 PCIe 16GB |
| CUDA Version     | 11.0                        |
| Driver Version   | 450.119                     |

## Documentation

### main.py

`main.py` trains networks that can transform black and white images into color images.

**Flags**

- `--train-type`: What type of model to train, we support a face recolor model and a general recolor model. Required
- `--save-path`: Path to save the models too. Required
- `--pretrain`: If to pre-train the gan, gives better results. Required
- `--epochs`: Number of times to train the model for. Required
- `--use-gpu`: If to use a GPU to train the model. Required

### infer.py

**Flags**

- `--checkpoint-dir`: Path to the trained model. Required
- `--input-path`: Path to image to recolor. Required
- `--output`: Path to save the generated image to. Required

## Attributions/Thanks

- [Image-to-Image Translation with Conditional Adversarial Networks paper](https://arxiv.org/abs/1611.07004), which you may know by the name pix2pix.
- [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) which tackled this problem like a classification problem but had its pros and cons.
