import argparse
from argparse import ArgumentParser

# from data.faces import download_faces_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_parser():
    parser = ArgumentParser()


    parser.add_argument('--data-path', type=str,
                        dest='data_path', help='Path to data',
                        metavar='IN_PATH', required=False)

    parser.add_argument('--save-path', type=str,
                        dest='save_path',
                        help='Path to save the model',
                        metavar='SAVE_PATH', required=True)


    help_out = 'If to pretrain the GAN before training the main model'
    parser.add_argument('--pretrain', type=str2bool,
                        dest='pretrain', help=help_out, metavar='PRETRAIN',
                        required=True)
    return parser

parser = build_parser()
opts = parser.parse_args()

print(opts)