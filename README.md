<h1 align="center"> Recolor</h1>
<h3 align="center">Retouch old black and white imges whith Recolor!</h3>

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
python3 recolor/main.py --train-type general \
  --save-path /kaggle/working/output \
  --pretrain y \
  --epochs 20 \
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
