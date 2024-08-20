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

DATASET_DEFINITION_IMAGE_ROOT = 'SeeTurtleID-MD+SAM-foreground'


def get_image_path(image_root, target):
    image_path = os.path.join(image_root, target.file_name)

    return image_path



class TurtleDataset(VisionDataset):

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

        annotation_file = os.path.join(root, 'metadata_base.csv')

        df = pd.read_csv(annotation_file)

        row_indexes = []
        for index, row in df.iterrows():
            image_path = get_image_path(os.path.join(root, DATASET_DEFINITION_IMAGE_ROOT), row)
            if not os.path.exists(image_path):
                warnings.warn(f'Path {image_path} does not exist')
            else:
                row_indexes.append(index)

        print('Using {} out of {} files'.format(len(row_indexes), len(df)))
        df = df.iloc[row_indexes]
        df = df.reset_index()

        df_identities = df[df.split_closed == 'train'].groupby(["identity"]).agg(
            count_col=pd.NamedAgg(column="identity", aggfunc="count")
        )
        df_identities = df_identities[df_identities.count_col > COUNT_THRESHOLD]
        identities = df_identities.index.tolist()

        df = df[df.identity.isin(identities)]

        self.name_to_identity_map = {v: ix for ix, v in enumerate(identities)}

        self.label_counts = np.zeros(len(identities))
        for index, identity in df_identities.iterrows():
            self.label_counts[self.name_to_identity_map[index]] = identity.count_col

        self.num_classes = len(self.name_to_identity_map)


        if split_name == 'closed':

            if train:
                df = df[(df.split_closed == 'train') | (df.split_closed == 'valid')]
            else:
                df = df[df.split_closed == 'test']

        else:
            assert False, 'Unknown split'


        df = df.reset_index()

        if train:
            identity_to_year_map = [{} for _ in range(len(identities))]
            for index, row in df.iterrows():
                identity_id = self.name_to_identity_map[row.identity]
                if row.year not in identity_to_year_map[identity_id]:
                    identity_to_year_map[identity_id][row.year] = 0

                identity_to_year_map[identity_id][row.year] += 1

            self.identity_to_last_year_map = []
            self.identity_to_all_years_map = identity_to_year_map
            for identity_id in range(len(identities)):
                years = list(sorted([(k, v) for k, v in identity_to_year_map[identity_id].items()], key=lambda x: -x[0]))
                self.identity_to_last_year_map.append(years[0][0])



        self.annotations = df
        self.labels = [self.name_to_identity_map[row.identity] for index, row in df.iterrows()]













    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:

        target = self.annotations.iloc[index, :]


        image = Image.open(get_image_path(self.image_root, target)).convert("RGB")
        # source = np.array(image)

        # tst = np.array(image)

        label = self.name_to_identity_map[target.identity]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, target.to_json()

    def __len__(self) -> int:
        return len(self.annotations)