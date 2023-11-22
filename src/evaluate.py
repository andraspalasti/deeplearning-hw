import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader: DataLoader, device: torch.device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', position=0, leave=False):
        images, true_masks = batch

        # move images and labels to correct device and type
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.float32)

        # predict the mask (shape: B x C x H x W)
        mask_preds = net(images)

        if net.n_classes == 1:
            assert true_masks.min() >= 0 and true_masks.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_preds = (F.sigmoid(mask_preds) > 0.5)
            # compute the Dice score
            dice_score += dice_coeff(mask_preds, true_masks, reduce_batch_first=False)
        else:
            assert true_masks.min() >= 0 and true_masks.max() < net.n_classes, 'True mask indices should be in [0, n_classes)'
            # convert to one-hot format
            mask_preds = F.one_hot(mask_preds.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2)
            # compute the Dice score, ignoring background
            dice_score += multiclass_dice_coeff(mask_preds[:, 1:], true_masks[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)


if __name__ == '__main__':
    from src.unet import UNet

    x, y = torch.rand((3, 256, 256)), torch.zeros((3, 256, 256))
    loader = DataLoader([(x, y)])
    net = UNet(n_channels=3, n_classes=3)
    evaluate(net, loader, torch.device('cpu'))
