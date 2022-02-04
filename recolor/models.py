import torch
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torch import nn
from torchvision.models.resnet import resnet18

from .modules import UnetBlock


class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(
            num_filters * 8, num_filters * 8, innermost=True)

        for _ in range(n_down - 5):
            unet_block = UnetBlock(
                num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8

        for _ in range(3):
            unet_block = UnetBlock(
                out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2

        self.model = UnetBlock(
            output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)


def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G
