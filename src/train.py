import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.evaluate import evaluate


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    epochs: int = 5,
):
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update({
        'epochs': epochs,
        'batch_size': train_loader.batch_size,
        'learning_rate': learning_rate,
    })

    # Set up optimizer, the loss
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, foreach=True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i, batch in enumerate(train_loader):
                images, true_masks = batch

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update()
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Log statistics
                if (i+1) % (len(train_loader) // 4) == 0:
                    val_score = evaluate(model, val_loader, device)
                    experiment.log({
                        'learning rate': learning_rate,
                        'validation Dice': val_score,
                        'images': wandb.Image(images[0].permute(1, 2, 0).cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].cpu()),
                            'pred': wandb.Image(masks_pred.argmax(dim=1)[0].cpu()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                    })