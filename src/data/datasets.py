from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# The size of the input image to the network
IMAGE_SIZE = 256


class AirbusDataset(Dataset):
    """Divides all images into 3x3 squares and loops through them."""

    def __init__(self, segmentations_file, imgs_dir, should_contain_ship=False):
        segmentations = pd.read_csv(segmentations_file)
        segmentations['EncodedPixels'] = segmentations['EncodedPixels'].fillna('')
        segmentations = segmentations[segmentations['EncodedPixels'] != '']

        self.should_contain_ship = should_contain_ship
        self.imgs_dir = Path(imgs_dir)
        if should_contain_ship:
            # Only use images that do contain a ship
            segmentations['ship_position'] = segmentations['EncodedPixels'].apply(self.ship_position)
            self.seg_by_img = segmentations.groupby('ImageId').aggregate({
                'EncodedPixels': ' '.join,
                'ship_position': lambda x: np.unique(np.concatenate(x.values))
            })

            # Create idx mapping to each image
            self.idx_to_dfi = []
            for i, positions in enumerate(self.seg_by_img['ship_position'].values):
                self.idx_to_dfi.extend([(i, p) for p in positions])
            self.num_images = len(self.idx_to_dfi)
        else:
            self.seg_by_img = segmentations.groupby(['ImageId']).agg({'EncodedPixels': ' '.join})
            self.num_images = len(self.seg_by_img)*3*3

    def ship_position(self, rle: str) -> list[int]:
        mask = decode_rle(rle, (768, 768))
        parts = mask.reshape(3, 256, 3, 256)
        return np.nonzero(parts.any(axis=(1, 3)).flatten())[0]

    def __len__(self):
        return self.num_images

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if self.should_contain_ship:
            dfi, pos = self.idx_to_dfi[index]
            img_id = self.seg_by_img.index.values[dfi]
            rle = self.seg_by_img.iloc[dfi, 0]
            ix, iy = pos % 3, pos // 3
        else:
            img_id = self.seg_by_img.index.values[index // 9]
            rle = self.seg_by_img.iloc[index // 9, 0]
            ix, iy = (index % 9) % 3, (index % 9) // 3

        # Which square of the image do we want to return
        starty, endy, startx, endx = IMAGE_SIZE*iy, IMAGE_SIZE*(iy+1), IMAGE_SIZE*ix, IMAGE_SIZE*(ix+1)

        img = Image.open(self.imgs_dir / img_id)
        crop = img.crop((startx, starty, endx, endy))
        input = transforms.F.to_tensor(crop)

        if rle == '': # There is no ship on the image
            return input, torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE)) 
        mask = decode_rle(rle, (img.width, img.height))
        mask = torch.from_numpy(mask[starty:endy, startx:endx])
        return input, mask.unsqueeze(dim=0)


class AirbusTrainingset(Dataset):
    def __init__(self, segmentations_file, imgs_dir):
        ship_segmentations = pd.read_csv(segmentations_file)
        ship_segmentations['EncodedPixels'] = ship_segmentations['EncodedPixels'].fillna('')

        self.ships = ship_segmentations
        self.seg_by_img = ship_segmentations.groupby(['ImageId']).agg({'EncodedPixels': ' '.join})
        self.imgs_dir = Path(imgs_dir)

    def __len__(self):
        return len(self.ships)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        img_id, rle = self.ships.iloc[index].tolist()
        img = Image.open(self.imgs_dir / img_id)
        if rle == '':
            # Create a random crop from the image
            starty, startx = torch.randint(low=0, high=img.width-IMAGE_SIZE, size=(2,)).tolist()
            crop = img.crop((startx, starty, startx+IMAGE_SIZE, starty+IMAGE_SIZE))
            mask = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE)) 
        else:
            # Create a crop around the ship
            ship_mask = decode_rle(rle, (img.width, img.height))
            ymin, ymax, xmin, xmax = bbox(ship_mask) # Bounding box of the ship
            bheight, bwidth = ymax-ymin, xmax-xmin

            # Add padding to values
            if bheight < 256:
                ymin = max(0, ymin - (256 - bheight)//2)
                ymax = min(img.height, ymax + (256 - bheight)//2)
            if bwidth < 256:
                xmin = max(0, xmin - (256 - bwidth)//2)
                xmax = min(img.width, xmax + (256 - bwidth)//2)

            crop = img.crop((xmin, ymin, xmax, ymax)).resize((IMAGE_SIZE, IMAGE_SIZE))
            mask = decode_rle(self.seg_by_img.loc[img_id, 'EncodedPixels'], (img.width, img.height))
            mask = Image.fromarray(mask)
            mask = mask.crop(((xmin, ymin, xmax, ymax))).resize((IMAGE_SIZE, IMAGE_SIZE))
            mask = transforms.F.to_tensor(mask)

        return transforms.F.to_tensor(crop), mask


def decode_rle(rle: str, img_size: tuple[int,int]) -> np.ndarray:
    """Decodes run length encoding into a mask that is the size of img_size.
    
    Args:
        rle (str): The run length encoding
        img_size (tuple[int, int]): The size of the original image (width, height)
    """
    rle = rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                        for x in (rle[0:][::2], rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(img_size[0]*img_size[0], dtype=np.bool_)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = True
    mask = mask.reshape((img_size[1], img_size[0]))
    return mask.transpose()


def bbox(mask: np.ndarray):
    """Returns the bounding box of the mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


if __name__ == '__main__':
    proc_dir = Path('data/processed')
    val_set = AirbusDataset(
        proc_dir / 'val_ship_segmentations.csv', 
        proc_dir / 'val',
        should_contain_ship=True
    )
    print(f'Length of dataset: {len(val_set)}')

    import matplotlib.pyplot as plt
    n = 3
    for i in range(n):
        crop, mask = val_set[3+i]
        plt.subplot(n, 2, i*2+1)
        plt.imshow(crop.permute((1, 2, 0)))
        plt.subplot(n, 2, i*2+2)
        plt.imshow(mask.squeeze())
    plt.tight_layout()
    plt.show()
