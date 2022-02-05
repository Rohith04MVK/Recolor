import argparse
import glob

import numpy as np
import torch
from data.faces import download_faces_data
from fastai.data.external import URLs, untar_data
from torch import nn, optim

from .data_loaders import make_dataloaders
from .models import MainModel, build_res_unet
from .train import pretrain_generator, train_model
from .utils import exists, str2bool


def check_opts(opts):
    # exists(opts.data_path, "Data path not found!")
    exists(opts.save_path, "Save path not found!")
    assert opts.epochs > 0, "Epochs must be higher than 0"


def validate_data_type(dtype):
    if dtype.lower() in ("face", ""):
        return dtype.lower()
    else:
        raise argparse.ArgumentTypeError("Invalid train type, supported are `face` and `general`")


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-type", type=validate_data_type,
                        dest="type", help="What type of model to train",
                        metavar="TRAIN_TYPE", required=True)

    parser.add_argument("--data-path", type=str,
                        dest="data_path", help="Path to data",
                        metavar="IN_PATH", required=False)

    parser.add_argument("--save-path", type=str,
                        dest="save_path",
                        help="Path to save the model",
                        metavar="SAVE_PATH", required=True)

    help_out = "If to pretrain the GAN before training the main model"
    parser.add_argument("--pretrain", type=str2bool,
                        dest="pretrain", help=help_out, metavar="PRETRAIN",
                        required=True)

    parser.add_argument("--epochs", type=int,
                        dest="epochs",
                        help="Number epochs to train the model for",
                        metavar="EPOCHS", required=True)

    parser.add_argument("--use-gpu", type=str2bool,
                        dest="use_gpu",
                        help="Use a GPU if available",
                        metavar="USE_GPU", required=True)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    check_opts(options)
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if options.use_gpu is False:
        device = "cpu"

    paths = []
    if options.type == "face":
        download_faces_data()
        paths.extend(glob.glob("~/datasets/celebahq/celeba_hq/train/female/*.jpg"))
        paths.extend(glob.glob("~/datasets/celebahq/celeba_hq/train/male/*.jpg"))
        paths.extend(glob.glob("~/datasets/celebahq/celeba_hq/val/female/*.jpg"))
        paths.extend(glob.glob("~/datasets/celebahq/celeba_hq/val/male/*.jpg"))
        print("Celebrity face dataset loaded!")
    else:
        coco_path = untar_data(URLs.COCO_SAMPLE)
        coco_path = str(coco_path) + "/train_sample"
        paths.extend(glob.glob(coco_path + "/*.jpg"))
        print("COCO dataset loaded!")

    np.random.seed(123)
    paths_subset = np.random.choice(paths, 20_000, replace=False)  # choosing 1000 images randomly
    rand_idxs = np.random.permutation(20_000)
    train_idxs = rand_idxs[:18_000]  # choosing the first 8000 as training set
    val_idxs = rand_idxs[18_000:]  # choosing last 2000 as validation set
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    print(len(train_paths), len(val_paths))

    train_dl = make_dataloaders(paths=train_paths, split='train')
    val_dl = make_dataloaders(paths=val_paths, split='val')

    data = next(iter(train_dl))
    Ls, abs_ = data['L'], data['ab']
    print(Ls.shape, abs_.shape)
    print(len(train_dl), len(val_dl))

    if options.pretrain is False:
        model = MainModel(device=device)
        train_model(model, train_dl, val_dl, options.epochs)
    else:
        net_G = build_res_unet(n_input=1, n_output=2, size=256, device=device)
        opt = optim.Adam(net_G.parameters(), lr=1e-4)
        criterion = nn.L1Loss()        
        pretrain_generator(net_G, train_dl, opt, criterion, 20, device=device)
        torch.save(net_G.state_dict(), "{options.save_path}/res18-unet.pt")

        net_G = build_res_unet(n_input=1, n_output=2, size=256)
        net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
        model = MainModel(net_G=net_G, device=device)
        train_model(model, train_dl, 20)
        torch.save(model.state_dict(), "final_model_weights.pt")
