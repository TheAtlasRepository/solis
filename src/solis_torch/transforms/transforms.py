#!/usr/bin/env python3
import torch
import torchvision.transforms.v2.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *imgs):
        for transform in self.transforms:
            imgs = transform(*imgs)
        return imgs


class ToTensor:
    def __call__(self, *imgs):
        return tuple(map(torch.from_numpy, imgs))


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, *imgs):
        return F.normalize(imgs[0], self.mean, self.std, False), *imgs[1:]


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *imgs):
        if torch.rand(1) < self.p:
            return tuple(map(F.hflip, imgs))
        return imgs


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, *imgs):
        if torch.rand(1) < self.p:
            return tuple(map(F.vflip, imgs))
        return imgs
