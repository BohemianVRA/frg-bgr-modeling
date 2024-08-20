import csv
import json
import os.path
import random
import warnings
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from PIL import Image
from torchvision.datasets import VisionDataset
from PIL import ImageFilter

from data_loader.data_split import split_train_val_instances


WIDTH_ENLARGE = 0.1
HEIGHT_ENLARGE = 0.1


TRAIN_TEST_DATE_THRESHOLD = '2020-01-01'
COUNT_THRESHOLD = 5

DATASET_DEFINITION_IMAGE_ROOT = 'LynxID-MD+SAM-foreground'


def get_image_path(image_root, target):
    extension = target['file_path'].split('.')[-1]
    image_path = os.path.join(image_root, target['image_id'].replace('LynxID/', '') + '.' + extension)

    return image_path



class LynxDataSet(VisionDataset):

    def __init__(
        self,
        root: str,
        split_name: str,
        crop_mode:str,
        image_root: str,
        train: Optional[bool] = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.image_root = os.path.join(root, image_root)
        self.train = train
        self.crop_mode = crop_mode
        self.date_column_name = 'date'

        annotation_file = os.path.join(root, 'LynxID-Duha-12-10-2023.csv')

        df = pd.read_csv(annotation_file)

        row_indexes = []
        for index, row in df.iterrows():
            image_path = get_image_path(os.path.join(root, DATASET_DEFINITION_IMAGE_ROOT), row)
            if not os.path.exists(image_path):
                continue
                # warnings.warn(f'Path {image_path} does not exist')
            else:
                row_indexes.append(index)


        df = df.iloc[row_indexes]
        df = df.reset_index()

        df_identities = df[df.date < TRAIN_TEST_DATE_THRESHOLD].groupby(["unique_name"]).agg(
            count_col=pd.NamedAgg(column="unique_name", aggfunc="count")
        )
        df_identities = df_identities[df_identities.count_col > COUNT_THRESHOLD]
        identities = df_identities.index.tolist()

        df = df[df.unique_name.isin(identities)]

        self.name_to_identity_map = {v: ix for ix, v in enumerate(identities)}

        self.label_counts = np.zeros(len(identities))
        for index, identity in df_identities.iterrows():
            self.label_counts[self.name_to_identity_map[index]] = identity.count_col

        self.num_classes = len(self.name_to_identity_map)

        if split_name == 'date-closedset':

            if train:
                df = df[df.date < TRAIN_TEST_DATE_THRESHOLD]
            else:
                df = df[df.date > TRAIN_TEST_DATE_THRESHOLD]

        else:
            assert False, 'Unknown split'


        df = df.reset_index()





        if train:
            identity_to_grid_map = [{} for _ in range(len(identities))]
            for index, row in df.iterrows():
                identity_id = self.name_to_identity_map[row.unique_name]
                if row.grid_code not in identity_to_grid_map[identity_id]:
                    identity_to_grid_map[identity_id][row.grid_code] = 0

                identity_to_grid_map[identity_id][row.grid_code] += 1

            self.identity_to_base_location_map = []
            self.identity_to_all_locations_map = identity_to_grid_map
            for identity_id in range(len(identities)):
                grid_codes = list(sorted([(k, v) for k, v in identity_to_grid_map[identity_id].items()], key=lambda x: -x[1]))
                self.identity_to_base_location_map.append(grid_codes)



        self.annotations = df
        self.labels = [self.name_to_identity_map[row.unique_name] for index, row in df.iterrows()]













    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:

        target = self.annotations.iloc[index, :].to_dict()
        target['image_path'] = get_image_path(self.image_root, target)

        image = Image.open(target['image_path']).convert("RGB")
        # source = np.array(image)

        if self.crop_mode.startswith('tight'):
            target_bbox = json.loads(target['bbox'])

            if self.train:
                # Randomly enlarge
                left, right = random.uniform(0, WIDTH_ENLARGE) * target_bbox[2], random.uniform(0, WIDTH_ENLARGE) * target_bbox[2]
                top, bottom = random.uniform(0, HEIGHT_ENLARGE) * target_bbox[3], random.uniform(0, HEIGHT_ENLARGE) * target_bbox[3]
            else:
                left, right = WIDTH_ENLARGE * 0.5 * target_bbox[2], WIDTH_ENLARGE * 0.5 * target_bbox[2]
                top, bottom = HEIGHT_ENLARGE * 0.5 * target_bbox[3], HEIGHT_ENLARGE * 0.5 * target_bbox[3]

            crop_box = (
                max(target_bbox[0] - left, 0),
                max(target_bbox[1] - top, 0),
                min(target_bbox[2] + target_bbox[0] + left + right, image.width),
                min(target_bbox[3] + target_bbox[1] + top + bottom, image.height)
            )

            target['crop_box'] = crop_box


            image = image.crop(crop_box)

            if self.crop_mode == 'tight_erosion5':
                image = image.filter(ImageFilter.MinFilter(5))
            elif self.crop_mode == 'tight_erosion3':
                image = image.filter(ImageFilter.MinFilter(3))
        elif self.crop_mode == 'nocrop':
            pass
        else:
            assert False, 'Invalid crop mode'

        # tst = np.array(image)

        label = self.name_to_identity_map[target['unique_name']]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, json.dumps(target)

    def __len__(self) -> int:
        return len(self.annotations)