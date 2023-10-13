import csv
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class AirbusDataset(Dataset):
    def __init__(self, segmentations_file, imgs_dir):
        segmentations = pd.read_csv(segmentations_file)
        segmentations['EncodedPixels'] = segmentations['EncodedPixels'].fillna('')
        self.img_segmentations = segmentations.groupby(['ImageId']).agg({'EncodedPixels': ' '.join})

        self.imgs_dir = Path(imgs_dir)

    def __len__(self):
        return len(self.img_segmentations)

    def __getitem__(self, index: int):
        img_id = self.img_segmentations.index[index]
        image = Image.open(self.imgs_dir / img_id)

        # The run length encoding of an image
        rle = self.img_segmentations.iloc[index, 0].split()

        # Creating image mask from encoding
        starts, lengths = [np.asarray(x, dtype=int)
                           for x in (rle[0:][::2], rle[1:][::2])]
        starts -= 1
        ends = starts + lengths
        mask = np.zeros(image.height*image.width, dtype=np.bool_)
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = True
        mask = mask.reshape((image.height, image.width))
        return image, mask.transpose()


if __name__ == '__main__':
    data_dir = Path(__file__).parents[2] / 'data' / 'raw'
    dataset = AirbusDataset(data_dir / 'train_ship_segmentations_v2.csv', data_dir / 'train_v2')
    for image, segmentation in dataset:
        print(image, segmentation)
        print()
