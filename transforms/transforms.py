# --------------------------------------------------------
# Dual Cross Attention
# Copyright (c) 2023 Gorkem Can Ates
# Licensed under The MIT License [see LICENSE for details]
# Written by Gorkem Can Ates (gca45@miami.edu)
# --------------------------------------------------------


from transforms.custom_transforms import *

class Transforms:
    def __init__(self,
                mode='torch',
                shape:tuple((int, int)) = (512, 512),
                transform:bool=True 
                ) -> None:
        train_transforms = []
        val_transforms = []
        if shape != (512, 512):
            train_transforms.extend([Resize(shape=shape, mode=mode)])
            val_transforms.extend([Resize(shape=shape, mode=mode)])

        if transform:

            train_transforms.extend(
                              [RandomRotation(angles=(-60, 60), p=0.2, mode=mode), 
                               RandomHorizontalFlip(p=0.2, mode=mode), 
                               RandomVerticalFlip(p=0.2, mode=mode),
                               ])
                               

        self.train_transform = Compose(transforms=train_transforms)
        self.val_transform = Compose(transforms=val_transforms)



