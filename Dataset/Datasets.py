__author__ = "Gorkem Can Ates"
__email__ = "gca45@miami.edu"

import os
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# import matplotlib
# matplotlib.use('tkagg')
class DATASET:
    def __init__(self,
                 im_path,
                 mask_path,
                 train_transform,
                 val_transform,
                 experiment='GlaS/',
                 num_classes=1,
                 shape=(512, 512),
                 shuffle=True,
                 debug=False, 
                 batch_size=16):
        self.train_transform = train_transform
        self.val_transform = val_transform

        if experiment == 'Kvasir/':
            train_ids, test_ids = train_test_split(np.arange(os.listdir(im_path).__len__()),
                                                test_size=0.2,
                                                random_state=42,
                                                shuffle=shuffle)            
            if debug:
                train_ids, test_ids = train_test_split(np.arange(int(batch_size * 2)),
                                                    test_size=0.5,
                                                    random_state=42,
                                                    shuffle=shuffle)

            self.train_dataset = Kvasir(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=train_ids,
                                    shape=shape,
                                    transform=self.train_transform
                                    )
            self.test_dataset = Kvasir(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=test_ids,
                                    shape=shape,
                                    transform=self.val_transform
                                    )
        elif experiment == 'CVC/':
            train_ids, test_ids = train_test_split(np.arange(os.listdir(im_path).__len__()),
                                                test_size=0.25,
                                                random_state=42,
                                                shuffle=shuffle)            
            if debug:
                train_ids, test_ids = train_test_split(np.arange(int(batch_size * 2)),
                                                    test_size=0.5,
                                                    random_state=42,
                                                    shuffle=shuffle)

            self.train_dataset = CVC(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=train_ids,
                                    shape=shape,
                                    transform=self.train_transform
                                    )
            self.test_dataset = CVC(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=test_ids,
                                    shape=shape,
                                    transform=self.val_transform
                                    )
        elif experiment == 'SYNAPS/':
            ldir = os.listdir(im_path)
            l = max([int(s.split('_', 1)[0]) for s in ldir]) + 1
            train_ids, test_ids = train_test_split(np.arange(l),
                                                test_size=0.4,
                                                random_state=42,
                                                shuffle=shuffle)             
            if debug:
                train_ids, test_ids = train_test_split(np.arange(2),
                                                    test_size=0.5,
                                                    random_state=42,
                                                    shuffle=shuffle)
            self.train_dataset = SYNAPS(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=train_ids,
                                    shape=shape,
                                    transform=self.train_transform
                                    )
            self.test_dataset = SYNAPS(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=test_ids,
                                    shape=shape,
                                    transform=self.val_transform
            )
        elif experiment == 'GlaS/':
            im_dir = os.listdir(im_path)
            train_ids = [i for i, d in enumerate(im_dir) if d[:5] == 'train']
            test_ids = [i for i, d in enumerate(im_dir) if d[:4] == 'test']
            
            if debug:
                train_ids, test_ids = train_test_split(np.arange(2),
                                                    test_size=0.5,
                                                    random_state=42,
                                                    shuffle=shuffle)

            self.train_dataset = GlaS(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=train_ids,
                                    shape=shape,
                                    transform=self.train_transform
                                    )
            self.test_dataset = GlaS(image_dir=im_path,
                                    mask_dir=mask_path,
                                    num_classes=num_classes,
                                    indexes=test_ids,
                                    shape=shape,
                                    transform=self.val_transform
            )
        elif experiment == 'MoNuSeg/':
            train_im_dir = im_path + 'train_images'
            test_im_dir = im_path + 'test_images'
            train_mask_dir = mask_path + 'train_masks'
            test_mask_dir = mask_path + 'test_masks'

                            
            self.train_dataset = MoNuSeg(image_dir=train_im_dir,
                                    mask_dir=train_mask_dir,
                                    num_classes=num_classes,
                                    shape=shape,
                                    transform=self.train_transform
                                    )
            self.test_dataset = MoNuSeg(image_dir=test_im_dir,
                                    mask_dir=test_mask_dir,
                                    num_classes=num_classes,
                                    shape=shape,
                                    transform=self.val_transform
            )
        else:
            raise Exception('Dataset not found.')

        print('Data load completed.')



class SYNAPS(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 indexes,
                 shape=(512, 512),
                 transform=None,
                 ):
        self.im_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        self.shape = shape
        im_list = os.listdir(image_dir)
        self.im_list = [im for im in im_list if int(im.split('_', 1)[0]) in indexes]
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item])
        image = np.load(img_path)
        mask = np.load(mask_path)

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            image = np.array(image, dtype=np.float32)
            mask = np.array(mask, dtype=np.float32)
            image, mask = self.ToTensor(image), self.ToTensor(mask)
            mask_out = torch.zeros_like(mask)
            labels = [8, 4, 3, 2, 6, 11, 1, 7]
            for i, label in enumerate(labels, start=1):
                mask_out[mask == label] = i
            mask[mask > self.num_classes-1] = 0
            mask[mask < 0] = 0
        return image, mask_out.long()


    def __len__(self):
        return len(self.im_list)


class GlaS(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 indexes,
                 shape=(512, 512),
                 transform=None,
                 ):
        self.im_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        self.shape = shape
        im_list = os.listdir(image_dir)
        self.im_list = [im_list[i] for i in indexes]        
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item][:-4] + '_anno.bmp')
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            image = np.array(image)
            mask = np.array(mask, dtype=np.float32)
            image, mask = self.ToTensor(image), self.ToTensor(mask)
            mask[mask > self.num_classes] = self.num_classes
            mask[mask < 0] = 0
        return image, mask.long()

    def __len__(self):
        return len(self.im_list)

class MoNuSeg(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 shape=(512, 512),
                 transform=None,
                 ):
        self.im_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        self.shape = shape
        self.ToTensor = transforms.ToTensor()
        self.im_list = os.listdir(self.im_dir)

    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item].replace('tif', 'png'))
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            image = np.array(image)
            mask = np.array(mask, dtype=np.float32)
            image, mask = self.ToTensor(image), self.ToTensor(mask)
            mask[mask > self.num_classes] = self.num_classes
            mask[mask < 0] = 0
        return image, mask.long()


    def __len__(self):
        return len(self.im_list)

class ISIC(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 shape=(512, 512),
                 transform=None,
                 ):
        self.im_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        self.shape = shape
        self.ToTensor = transforms.ToTensor()
        self.im_list = os.listdir(self.im_dir)
        random.shuffle(self.im_list)

    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item].replace('.jpg', '_segmentation.png'))
        image = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            image = np.array(image)
            mask = np.array(mask, dtype=np.float32)
            image, mask = self.ToTensor(image), self.ToTensor(mask)
            mask[mask > self.num_classes] = self.num_classes
            mask[mask < 0] = 0
        return image, mask.long()


    def __len__(self):
        return len(self.im_list)

class Kvasir(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 indexes,
                 shape=(512, 512),
                 transform=None,
                 ):
        self.im_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        self.shape = shape
        im_list = os.listdir(image_dir)
        self.im_list = [im_list[i] for i in indexes]
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            image = np.array(image)
            mask = np.array(mask, dtype=np.float32)
            image, mask = self.ToTensor(image), self.ToTensor(mask)
            if self.num_classes == 1:
                mask[mask >= self.num_classes] = self.num_classes
            else:
                mask[mask > self.num_classes-1] = self.num_classes-1
            mask[mask < 0] = 0
        return image, mask.long()


    def __len__(self):
        return len(self.im_list)

class CVC(Dataset):
    def __init__(self,
                 image_dir,
                 mask_dir,
                 num_classes,
                 indexes,
                 shape=(512, 512),
                 transform=None,
                 ):
        self.im_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.num_classes = num_classes
        self.shape = shape
        im_list = os.listdir(image_dir)
        self.im_list = [im_list[i] for i in indexes]
        self.ToTensor = transforms.ToTensor()

    def __getitem__(self, item):
        img_path = os.path.join(self.im_dir, self.im_list[item])
        mask_path = os.path.join(self.mask_dir, self.im_list[item])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
            image = np.array(image)
            mask = np.array(mask, dtype=np.float32)
            image, mask = self.ToTensor(image), self.ToTensor(mask)
            if self.num_classes == 1:
                mask[mask >= self.num_classes] = self.num_classes
            else:
                mask[mask > self.num_classes-1] = self.num_classes-1
            mask[mask < 0] = 0
        return image, mask.long()


    def __len__(self):
        return len(self.im_list)


    

