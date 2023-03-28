# --------------------------------------------------------
# Dual Cross Attention
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------



from PIL import Image
from typing import Any
import numpy as np
import torchvision.transforms.functional as TF
import cv2
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

def plot(d):
    plt.figure()
    plt.imshow(d)

class Compose:
    def __init__(self, transforms:list) -> None:
        self.transforms = transforms

    def __call__(self, data, target):
        for transform in self.transforms:
            data, target = transform(data, target)
        return data, target


class Resize:
    def __init__(self, shape: tuple((int, int)), mode='torch') -> None:
        self.shape = shape
        self.mode = mode
    def __call__(self, data, target) -> Any:
        if self.mode == 'torch':
            data, target = TF.resize(data, self.shape, interpolation=Image.BILINEAR), TF.resize(target, self.shape, interpolation=Image.NEAREST)
        elif self.mode == 'cv2':
            data = cv2.resize(data, 
                                self.shape, 
                                interpolation=cv2.INTER_LINEAR)
            target = cv2.resize(target, 
                                self.shape, 
                                interpolation=cv2.INTER_NEAREST)
        else:
            raise Exception('Transform mode not found.')
        return data, target
        

class RandomRotation:
    def __init__(self, angles:tuple((int, int))=(-60, 60), p=0.5, mode='torch') -> None:
        self.angles = angles
        self.p = p
        self.mode = mode

    def __call__(self, data, target):
        if random.random() <= self.p:
            angle = random.randint(self.angles[0], self.angles[1])
            if self.mode == 'torch':
                data, target = TF.rotate(data, angle), TF.rotate(target, angle)
            elif self.mode == 'cv2':
                Md = cv2.getRotationMatrix2D((data.shape[0]/2, data.shape[1]/2), angle, 1)                 
                Mt = cv2.getRotationMatrix2D((target.shape[0]/2, target.shape[1]/2), angle, 1)                                 
                data = cv2.warpAffine(data,Md,(data.shape[0],data.shape[1])) 
                target = cv2.warpAffine(target,Mt,(target.shape[0], target.shape[1])) 
            else:
                raise Exception('Transform mode not found.')

        return data, target
        


class RandomHorizontalFlip:
    def __init__(self, p=0.5, mode='torch') -> None:
        self.p = p
        self.mode = mode

    def __call__(self, data, target):
        if random.random() <= self.p:
            if self.mode == 'torch':
                data, target = TF.hflip(data), TF.vflip(target)
            elif self.mode == 'cv2':
                data, target = cv2.flip(data, 1), cv2.flip(target, 1)
            else:
                raise Exception('Transform mode not found.')        
        return data, target

        
class RandomVerticalFlip:
    def __init__(self, p=0.5, mode='torch') -> None:
        self.p = p
        self.mode = mode

    def __call__(self, data, target):
        if random.random() <= self.p:
            if self.mode == 'torch':
                data, target = TF.vflip(data), TF.vflip(target)
            elif self.mode == 'cv2':
                data, target = cv2.flip(data, 0), cv2.flip(target, 0)
            else:
                raise Exception('Transform mode not found.')                 
        return data, target


class GrayScale:
    def __init__(self, p=0.5) -> None:
        self.p = p
    def __call__(self, data, target) -> Any:
        if random.random() <= self.p:
            data = TF.to_grayscale(data, num_output_channels=3)
        return data, target

class GaussianBlur:
    def __init__(self, kernel_size:list((int, int))=[11, 11], p=0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p
    def __call__(self, data, target) -> Any:
        if random.random() <= self.p:
            data = TF.gaussian_blur(data, kernel_size=self.kernel_size)
        return data, target


class ToTensor:
    def __call__(self, data, target):
        data = np.array(data)
        target = np.array(target, dtype=np.float32)
        data = transforms.ToTensor()(data)
        target = transforms.ToTensor()(target)
        return data, target

        
