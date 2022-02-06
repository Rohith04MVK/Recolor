import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SIZE = 256


# RGB2Lab transform
class RGB2Lab(object):
    def __call__(self, img):
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
        return img_lab


class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),  # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE), Image.BICUBIC)

        # Add RGBLab and ToTensor transforms
        self.transforms.transforms.append(RGB2Lab())
        self.transforms.transforms.append(transforms.ToTensor())

        self.split = split
        self.size = SIZE
        self.paths = paths

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        L = img[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = img[[1, 2], ...] / 110.  # Between -1 and 1

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)


# A handy function to make our dataloaders
def make_dataloaders(batch_size=16, pin_memory=True, **kwargs):
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory
    )
    return dataloader
