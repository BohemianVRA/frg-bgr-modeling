import os
from functools import partial
import torchvision.transforms.functional as F

import numpy as np
from torchvision import datasets, transforms

from base import BaseDataLoader


from data_loader.lynx_dataset import LynxDataSet
from data_loader.turtle_dataset import TurtleDataset


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


class LynxDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, training_mode, split_name, crop_mode, image_root, num_workers=1, training=True):

        if training:
            ts = transforms.Compose([
                SquarePad(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, saturation=0.1, hue=0.1, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            ts = transforms.Compose([
                SquarePad(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.data_dir = data_dir
        self.training_mode = training_mode

        self.dataset = LynxDataSet(self.data_dir, split_name, crop_mode, image_root, train=training, transforms=ts)


        self.num_classes = self.dataset.num_classes

        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)


class TurtleDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle, training_mode, split_name, crop_mode, image_root, num_workers=1, training=True):

        if training:
            ts = transforms.Compose([
                SquarePad(),
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, saturation=0.1, hue=0.1, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            ts = transforms.Compose([
                SquarePad(),
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.data_dir = data_dir
        self.training_mode = training_mode

        self.dataset = TurtleDataset(self.data_dir, split_name, crop_mode, image_root, train=training, transforms=ts)

        self.num_classes = self.dataset.num_classes

        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)

