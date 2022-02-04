import argparse
import os

from .utils import str2bool, exists

# from data.faces import download_faces_data


def check_opts(opts):
    exists(opts.checkpoint_dir, "Checkpoint not found!")
    exists(opts.in_path, "In path not found!")
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, "out dir not found!")
        assert opts.batch_size > 0


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--type", type=str, dest="type", help="What type of model to train", metavar="TYPE", required=False)

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
    return parser


def main():
    pass


if __name__ == "__main__":
    pass
