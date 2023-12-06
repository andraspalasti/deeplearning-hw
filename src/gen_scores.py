from pathlib import Path

import torch
from tqdm import tqdm
import pandas as pd

from src.unet import UNet
from src.evaluate import predict
from src.iou_score import IoU_score
from src.dice_score import dice_coeff
from src.data.datasets import AirbusDataset, AirbusRawDataset


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the UNet on test images and target masks')
    parser.add_argument('--load', '-f', type=str, required=True, help='Load model from a .pth file')
    parser.add_argument('--out', '-o', type=str, required=True, help='Where to save evaluation results')
    parser.add_argument('--split', action='store_true', default=False, help='Split images into 3x3 equal parts')
    return parser.parse_args()


def main():
    args = get_args()
    checkpoint_path = Path(args.load)
    out_path = Path(args.out)

    imgs_dir = Path('data/processed/test')
    segmentations_file = Path('data/processed/test_ship_segmentations.csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    state_dict = torch.load(str(checkpoint_path), map_location='cpu')
    del state_dict['learning_rate']
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f'Loaded model weights from: {checkpoint_path}')

    if args.split:
        test_set = AirbusDataset(
            imgs_dir=imgs_dir,
            segmentations_file=segmentations_file,
            should_contain_ship=False
        )
        image_ids = test_set.seg_by_img.index.tolist()
        parts = [0] * len(test_set)
    else:
        test_set = AirbusRawDataset(
            imgs_dir=imgs_dir,
            segmentations_file=segmentations_file,
        )
        image_ids = [test_set.seg_by_img.index[i/9] for i in range(len(test_set))]
        parts = [i%9 for i in range(len(test_set))]

    iou_scores, dice_scores = [], []
    with tqdm(desc='Evaluating', total=len(test_set)) as pbar:
        running_iou = 0
        running_dice = 0
        for i, (input_img, true_mask) in enumerate(test_set):
            input_img = input_img.to(device, non_blocking=True)
            true_mask = true_mask.to(device, non_blocking=True)

            mask_pred = predict(model, input_img)
            mask_pred = mask_pred.squeeze(dim=0)

            iou = IoU_score(mask_pred, true_mask).item()
            running_iou += iou
            iou_scores.append(iou)

            dice = dice_coeff(mask_pred, true_mask).item()
            running_dice += dice
            dice_scores.append(dice)

            pbar.set_postfix({
                'Running IoU': running_iou / (i+1), 
                'Running Dice': running_dice / (i+1)
            })
            pbar.update()

    evaluation = pd.DataFrame({
        'ImageId': image_ids,
        'part': parts,
        'iou': iou_scores,
        'dice': dice_scores,
    })
    evaluation.to_csv(out_path, sep=';')
    print(f'Successfully saved evaluation to: {out_path}')


if __name__ == '__main__':
    main()
