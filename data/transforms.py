"""Multi-frame consistent video transforms."""
import random
from typing import List
import torch
import torchvision.transforms.functional as TF
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class Resize:
    def __init__(self, size: int): self.size = size
    def __call__(self, frames): return [TF.resize(f, self.size) for f in frames]


class RandomResizedCrop:
    def __init__(self, size, scale=(0.08,1.0), ratio=(3/4,4/3)):
        self.size = size; self.scale = scale; self.ratio = ratio
    def __call__(self, frames):
        import torchvision.transforms as T
        i,j,h,w = T.RandomResizedCrop.get_params(frames[0], self.scale, self.ratio)
        out_size = [self.size, self.size] if isinstance(self.size, int) else self.size
        return [TF.resized_crop(f,i,j,h,w,out_size) for f in frames]


class CenterCrop:
    def __init__(self, size): self.size = size
    def __call__(self, frames): return [TF.center_crop(f, self.size) for f in frames]


class RandomHorizontalFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, frames):
        return [TF.hflip(f) for f in frames] if random.random() < self.p else frames


class ToTensorAndNormalize:
    def __init__(self, mean=MEAN, std=STD): self.mean=mean; self.std=std
    def __call__(self, frames):
        tensors = [TF.normalize(TF.to_tensor(f), self.mean, self.std) for f in frames]
        return torch.stack(tensors, dim=1)  # (C, T, H, W)


class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, frames):
        for t in self.transforms: frames = t(frames)
        return frames


def build_train_transforms(img_size=224):
    return Compose([Resize(256), RandomResizedCrop(img_size),
                    RandomHorizontalFlip(0.5), ToTensorAndNormalize()])


def build_val_transforms(img_size=224):
    return Compose([Resize(256), CenterCrop(img_size), ToTensorAndNormalize()])
