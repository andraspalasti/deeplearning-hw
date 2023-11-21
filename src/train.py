from pathlib import Path

import wandb
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluate import evaluate


class_labels = {0: 'background', 1: 'ship'}

def train_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    epochs: int = 5,
    checkpoint_dir = Path('checkpoint'),
):
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update({
        'epochs': epochs,
        'batch_size': train_loader.batch_size,
        'learning_rate': learning_rate,
    })

    print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {train_loader.batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Device:          {device.type}
    ''')

    # Set up optimizer, the loss
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, foreach=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
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

                masks_pred = model(images)

                # TODO: Only works on single class
                loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

                    # TODO: Only works for n_classes = 1
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
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        torch.save(state_dict, str(checkpoint_dir / f'checkpoint_epoch{epoch}.pth'))
        print(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    from src.unet import UNet
    from data.datasets import AirbusTrainingset

    data_dir = Path('data/processed')
    training_set = AirbusTrainingset(data_dir / 'train_ship_segmentations.csv', data_dir / 'train')
    train_loader = DataLoader(training_set, batch_size=4, shuffle=False)

    model = UNet(n_channels=3, n_classes=1)

    for images, true_masks in train_loader: break
    masks_pred = model(images)
    
    ixs = torch.nonzero(true_masks)[:, 0] # Indexes of images that contain ships
    image_ix = ixs[0].item() if ixs.size(0) > 0 else 0

    predicted_mask = (torch.squeeze(masks_pred[image_ix]) >= 0.5).type(torch.uint8).numpy()
    ground_truth = torch.squeeze(true_masks[image_ix]).numpy()

    image = wandb.Image(
        to_pil_image(images[image_ix]), 
        masks={
            'prediction': {'mask_data': predicted_mask, 'class_labels': class_labels},
            'ground_truth': {'mask_data': ground_truth, 'class_labels': class_labels}
        },
    )


