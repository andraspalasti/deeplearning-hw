from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import wandb
from src.dice_score import dice_loss
from src.evaluate import evaluate, predict_image

class_labels = {0: 'background', 1: 'ship'}

def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    epochs: int = 5,
    amp: bool = False,
    checkpoint_dir = Path('checkpoints'),
):
    assert model.n_classes == 1, 'Can only train binary classification model with this function'

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update({
        'epochs': epochs,
        'batch_size': train_loader.batch_size,
        'learning_rate': learning_rate,
        'amp': amp,
    })

    print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {train_loader.batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    grad_clipping = 1.0
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp) # Needed because of autocasting
    criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i, batch in enumerate(train_loader):
                images, true_masks = batch

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                # Clear gradients left by previous batch
                optimizer.zero_grad(set_to_none=True) # For reduced memory footprint

                # Forward pass
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred_logits = model(images)
                    loss = criterion(masks_pred_logits, true_masks)

                    # Predictions with probabilities
                    masks_pred = F.sigmoid(masks_pred_logits).squeeze(dim=1)
                    loss += dice_loss(
                        masks_pred.float(),
                        torch.squeeze(true_masks, dim=1).float(),
                        multiclass=False
                    )

                grad_scaler.scale(loss).backward() # Populate gradients
                nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
                grad_scaler.step(optimizer) # Do optimization step
                grad_scaler.update()

                # Update statistics
                pbar.update(train_loader.batch_size)
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Log statistics
                if (i+1) % (len(train_loader) // 10) == 0:
                    val_score = evaluate(model, val_loader, device)
                    print(f'Validation Dice score: {val_score}')

                    # Reduce lr when validation does not get better
                    scheduler.step(val_score)

                    # Search for an image containing a ship in batch
                    ixs = torch.nonzero(true_masks)[:, 0] # Indexes of images that contain ships
                    image_ix = ixs[0].item() if ixs.size(0) > 0 else 0
                    predicted_mask = (masks_pred[image_ix] > 0.5).cpu()
                    ground_truth = torch.squeeze(true_masks[image_ix], dim=0).cpu()

                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation Dice': val_score,
                        'image': wandb.Image(
                            to_pil_image(images[image_ix]), 
                            masks={
                                'prediction': {'mask_data': predicted_mask.numpy(), 'class_labels': class_labels},
                                'ground_truth': {'mask_data': ground_truth.numpy(), 'class_labels': class_labels}
                            },
                        ),
                        'masks': {
                            'pred': wandb.Image(predicted_mask.float()),
                            'true': wandb.Image(ground_truth.float()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                    })

        # Save checkpoint
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        save_model(model, epoch, optimizer.param_groups[0]['lr'], checkpoint_dir)
        print(f'Checkpoint {epoch} saved!')


def save_model(model: nn.Module, epoch: int, lr: float, dir):
    dir = Path(dir)
    state_dict = model.state_dict()
    state_dict['learning_rate'] = lr
    torch.save(state_dict, str(dir / f'checkpoint_epoch{epoch}.pth'))

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--predict', type=str, default=False, help='Load image and perform prediction on it (use with load flag to use trained model)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--eval', action='store_true', default=False, help='Only evaluate the model')
    return parser.parse_args()


def main():
    import os

    from data.datasets import AirbusDataset, AirbusTrainingset
    from src.unet import UNet

    args = get_args()

    data_dir = Path('data/processed')
    training_set = AirbusTrainingset(data_dir / 'train_ship_segmentations.csv', data_dir / 'train')
    test_set = AirbusDataset(data_dir / 'val_ship_segmentations.csv', data_dir / 'val')
    validation_set = AirbusDataset(
        data_dir / 'val_ship_segmentations.csv',
        data_dir / 'val',
        should_contain_ship=True
    )

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True, 
        generator=torch.Generator().manual_seed(42) # So we have the same shuffling through each training
    )
    train_loader = DataLoader(training_set, shuffle=True, **loader_args)
    val_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=1, bilinear=args.bilinear)
    model.to(device)
    print(f'Network:\n'
        f'\t{model.n_channels} input channels\n'
        f'\t{model.n_classes} output channels (classes)\n'
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    learning_rate = args.lr
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'learning_rate' in state_dict:
            learning_rate = state_dict['learning_rate']
            del state_dict['learning_rate']
        model.load_state_dict(state_dict)
        print(f'Successfully loaded model from {args.load}')

    if args.predict:
        out_img, out_mask = Path('colored.png'), Path('mask.png')
        print(f'Segmenting image: {args.predict}')
        mask = predict_image(model, args.predict, out_img)
        Image.fromarray(mask).save(out_mask)
        print(f'Colored image saved to: {out_img}')
        print(f'Predicted mask saved to: {out_mask}')
        return

    if args.eval:
        dice_score = evaluate(model, test_loader, device)
        print(f'Validation Dice: {dice_score}')
        return

    try:
        train_model(
            model,
            device,
            train_loader,
            val_loader,
            learning_rate=learning_rate,
            epochs=args.epochs,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print('Detected OutOfMemoryError!')


if __name__ == '__main__':
    main()
