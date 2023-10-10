import csv
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class AirbusDataset(Dataset):
    # TODO: There is a mistake in this because multiple rows in the csv have the same ImageId
    # because multiple ships can be on a single image
    def __init__(self, segmentations_file, imgs_dir):
        self.img_segmentations: list[Tuple[str, np.ndarray]] = []
        with open(segmentations_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.img_segmentations.append((
                    row['ImageId'],
                    np.array(row['EncodedPixels'].split(), dtype=int).reshape((-1, 2))
                ))

        self.imgs_dir = Path(imgs_dir)

    def __len__(self):
        return len(self.img_segmentations)

    def __getitem__(self, index: int):
        img_id, segmentation = self.img_segmentations[index]
        image = Image.open(self.imgs_dir / img_id)
        return image, segmentation


if __name__ == '__main__':
    data_dir = Path(__file__).parents[2] / 'data' / 'raw'
    dataset = AirbusDataset(data_dir / 'train_ship_segmentations_v2.csv', data_dir / 'train_v2')
    for image, segmentation in dataset:
        print(image, segmentation)
        print()
