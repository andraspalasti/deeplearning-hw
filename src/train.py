from pathlib import Path

import wandb
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluate import evaluate
from src.dice_score import dice_loss


class_labels = {0: 'background', 1: 'ship'}

def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    epochs: int = 5,
    amp: bool = False,
    checkpoint_dir = Path('checkpoint'),
):
    assert model.n_classes == 1, 'Can binary classification model with this function'

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
    # Create folder for checkpoints
    if not isinstance(checkpoint_dir, Path): checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    grad_clipping = 1.0
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, foreach=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp) # Needed because of autocasting
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

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

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    loss += dice_loss(
                        nn.functional.sigmoid(masks_pred).squeeze(dim=1).float(),
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
                if (i+1) % (len(train_loader) // 5) == 0:
                    val_score = evaluate(model, val_loader, device)
                    print(f'Validation Dice score: {val_score}')

                    # Search for an image containing a ship in batch
                    ixs = torch.nonzero(true_masks)[:, 0] # Indexes of images that contain ships
                    image_ix = ixs[0].item() if ixs.size(0) > 0 else 0
                    predicted_mask = (torch.squeeze(masks_pred[image_ix]) >= 0.5).cpu().numpy()
                    ground_truth = torch.squeeze(true_masks[image_ix]).cpu().numpy()

                    experiment.log({
                        'learning rate': learning_rate,
                        'validation Dice': val_score,
                        'image': wandb.Image(
                            to_pil_image(images[image_ix]), 
                            masks={
                                'prediction': {'mask_data': predicted_mask, 'class_labels': class_labels},
                                'ground_truth': {'mask_data': ground_truth, 'class_labels': class_labels}
                            },
                        ),
                        'step': global_step,
                        'epoch': epoch,
                    })

        # Save checkpoint
        save_model(model, epoch, i, learning_rate, checkpoint_dir)
        print(f'Checkpoint {epoch} saved!')


def save_model(model: nn.Module, epoch: int, lr: float, dir):
    dir = Path(dir)
    state_dict = model.state_dict()
    state_dict['epoch'] = epoch
    state_dict['learning_rate'] = lr
    torch.save(state_dict, str(dir / f'checkpoint_epoch{epoch}.pth'))


if __name__ == '__main__':
    from src.unet import UNet
    from data.datasets import AirbusTrainingset, AirbusDataset

    data_dir = Path('data/processed')
    training_set = AirbusTrainingset(data_dir / 'train_ship_segmentations.csv', data_dir / 'train')
    validation_set = AirbusDataset(
        data_dir / 'val_ship_segmentations.csv',
        data_dir / 'val',
        should_contain_ship=True
    )

    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(training_set, batch_size=1, shuffle=True, pin_memory=True, generator=g)
    val_loader = DataLoader(validation_set, batch_size=1, shuffle=False, pin_memory=True, generator=g)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=1)
    print(f'Network:\n'
        f'\t{model.n_channels} input channels\n'
        f'\t{model.n_classes} output channels (classes)\n'
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    model.to(device)
    try:
        train_model(
            model,
            device,
            train_loader,
            val_loader,
            learning_rate=0.0001,
            epochs=5,
            amp=False
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print('Detected OutOfMemoryError!')
