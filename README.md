<h1 align="center"> Recolor</h1>
<h3 align="center">Retouch old black and white imges whith Recolor!</h3>

## Installation and Usage

This project uses `pipenv` for dependency management. You need to ensure that you have pipenv installed.

Here are the commands to facilitate using this project.

#### Clone the repo

```sh
git clone https://github.com/Rohith04MVK/Recolor
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

#### Run the main script

```sh
python example.py
```
