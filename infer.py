import argparse

import numpy as np
import PIL
import torch
from torchvision import transforms

from recolor.models import MainModel, build_res_unet
from recolor.utils import exists, lab_to_rgb, str2bool


def check_opts(opts):
    exists(f"{opts.checkpoint_dir}/res18-unet.pt", "Resunet is missing or has a different name")
    exists(f"{opts.checkpoint_dir}/final_model_weights.pt", "Final model is missing or has a different name")


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        dest="checkpoint_dir",
        help="Path to data",
        metavar="IN_PATH",
        required=False
    )

    parser.add_argument(
        "--input-image",
        type=str,
        dest="input_path",
        help="Path to the image to process",
        metavar="INPUT_PATH",
        required=True
    )

    parser.add_argument(
        "--output",
        type=str,
        dest="output_path",
        help="Path to save the image",
        metavar="OUTPUT_PATH",
        required=True
    )

    parser.add_argument(
        "--use-gpu",
        type=str2bool,
        dest="use_gpu",
        help="Use a GPU if available",
        metavar="USE_GPU",
        required=True
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    options = parser.parse_args()

    check_opts(options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if options.use_gpu is False:
        device = "cpu"

    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load(f"{options.checkpoint_dir}/res18-unet.pt", map_location=device))
    model = MainModel(net_G=net_G)
    model.load_state_dict(torch.load(f"{options.checkpoint_dir}/final_model_weights.pt", map_location=device))

    img = PIL.Image.open(options.input_path)
    img = img.resize((256, 256))
    # to make it between -1 and 1
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    result = PIL.Image.fromarray((colorized * 255).astype(np.uint8))
    result.save(options.output_path)
