import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dice_score import dice_coeff, multiclass_dice_coeff


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


@torch.inference_mode()
def predict(net, inputs: torch.Tensor) -> torch.Tensor:
    if len(inputs.shape) == 3:
        inputs = inputs.unsqueeze(dim=0)
    net.eval()
    out = net(inputs)
    mask_preds = (F.sigmoid(out) > 0.5)
    return mask_preds


def predict_image(net, img_path, out_path):
    size = 256 # The size of the input images to the network
    img = Image.open(img_path)
    if img.width < size:
        ratio = size / img.width
        img = img.resize((size, int(ratio * img.height)))
    if img.height < size:
        ratio = size / img.height
        img = img.resize((int(ratio * img.width), size))

    w, h = img.width, img.height
    img = np.array(img)

    # Split image into regions
    xs = [x for x in range(0, w-size, size)] + [w - size]
    ys = [y for y in range(0, h-size, size)] + [h - size]
    crops = [torch.from_numpy(img[y:y+size, x:x+size]) for y in ys for x in xs]

    # Create inputs for network
    inputs = torch.cat([torch.unsqueeze(crop, dim=0) for crop in crops], dim=0)
    inputs = inputs.permute((0, 3, 1, 2)).type(torch.float)
    inputs /= 255

    mask_preds = predict(net, inputs)

    # Create mask for whole image
    mask = np.empty((h, w), dtype=np.bool_)
    for i, crop_mask in enumerate(mask_preds):
        y, x = ys[i//len(xs)], xs[i%len(xs)]
        mask[y:y+size, x:x+size] = crop_mask.numpy()

    # Merge the mask and the original image
    merged = np.ones((h, w, 4))*255
    merged[:,:,:-1] = img
    merged[mask, :-1] = merged[mask, :-1]*0.6 + np.array([255, 0, 0])*0.4
    merged = merged.astype(np.uint8)

    out_img = Image.fromarray(merged)
    out_img.save(out_path)
    return mask


if __name__ == '__main__':
    from src.unet import UNet

    x, y = torch.rand((3, 256, 256)), torch.zeros((3, 256, 256))
    loader = DataLoader([(x, y)])
    net = UNet(n_channels=3, n_classes=3)
    evaluate(net, loader, torch.device('cpu'))
