import torch
import torch.nn.functional as F
import torch.optim as optim
import time

import matplotlib.pyplot as plt
import numpy as np

import modules
import dataset

def soft_dice_loss(data, mask, eps=1e-6):
    """
    Uses probability values to calculate loss for each slice in the batch.
    data (tensor): [B, C, H, W] shape containing the model's predictions
    mask (tensor): [B, C, H, W] shape containing the true segmentation mask
    eps (Float): used for gradient stabilisation
    """
    probabilities = torch.sigmoid(data)

    # flatten to [B, C, HxW]
    pred = probabilities.view(probabilities.size(0), probabilities.size(1), -1)
    target = mask.view(mask.size(0), mask.size(1), -1)

    intersection = (pred * target).sum(dim=(0, 2)) #|X int Y|
    total = pred.sum(dim=(0, 2)) + target.sum(dim=(0, 2)) #|X| + |Y|

    # +eps is stabilisation for empty slices
    channel_dice = (2.0 * intersection + eps) / (total + eps)
    return 1.0 - channel_dice.mean()

def loss(data, mask, bce_weight=0.5):
    """Combination of BCE and dice loss used for training and validation.

    data (tensor): [B, C, H, W] shape containing the model's predictions
    mask (tensor): [B, C, H, W] shape containing the true segmentation mask
    bce_weight (Float): BCE's weighting in the overall loss function
    """
    # take sigmoid of data and compute BCE between data and mask
    bce_loss = F.binary_cross_entropy_with_logits(data, mask)
    # take sigmoid of data and compute soft dice loss for the batch of slices
    soft_dice = soft_dice_loss(data, mask)

    return bce_loss * bce_weight + (1-bce_weight) * soft_dice

def hard_dice_loss(data, mask, threshold=0.5):
    """
    Uses prediction results to calculate loss for each channel.
    data (tensor): [B, C, H, W] shape containing the model's predictions
    mask (tensor): [B, C, H, W] shape containing the true segmentation mask
    eps (Float): used for gradient stabilisation
    """
    with torch.no_grad():
        probabilities = torch.sigmoid(data) # convert values to [0, 1]
        predictions = (probabilities >= threshold).float() # convert values to {0, 1}

        # flatten to [B, C, HxW]
        pred = predictions.view(predictions.size(0), predictions.size(1), -1)
        target = mask.view(mask.size(0), mask.size(1), -1)

        intersection = (pred * target).sum(dim=2) #|X int Y|
        total = pred.sum(dim=2) + target.sum(dim=2) #|X| + |Y|

        return intersection, total # maintain array structure

def train_epoch(model, dataloader, device, optimiser):
    """
    Execute one epoch of training for the given model.
    """
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        optimiser.zero_grad()
        probabilities = model(images)

        # loss per batch
        batch_loss = loss(probabilities, masks)
        batch_loss.backward()
        optimiser.step()

        # extract loss value, scale by batch size
        running_loss += batch_loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset) # loss per slice

def validate_epoch(model, dataloader, device):
    """
    Evaluates a model prediction and compares to the ground truth with dice loss
    """
    model.eval()
    running_loss = 0.0
    total_intersection = torch.zeros(5, device=device)
    total_pixels = torch.zeros(5, device=device)

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            probabilities = model(images)
            # loss for the current batch
            batch_loss = loss(probabilities, masks)
            running_loss += batch_loss.item() * images.size(0)
            intersection, total = hard_dice_loss(probabilities, masks)
            # updte total loss values
            total_intersection += intersection.sum(dim=0)
            total_pixels += total.sum(dim=0)

    val_loss = running_loss / len(dataloader.dataset)
    eps = 1e-6 # eps for gradient stablisation
    all_channel_dice = (2.0 * total_intersection + eps) / (total_pixels + eps)
    prostate_dice = all_channel_dice[4].item()

    return all_channel_dice, prostate_dice, val_loss

def plot_loss_curves(train_loss, val_loss, num_epochs):
    """
    Plots the training loss and validation loss over each epoch.
    """
    e = range(1, num_epochs+1)

    plt.figure()
    plt.plot(e, train_loss, label="Training loss", color="blue")
    plt.plot(e, val_loss, label="Validation loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Average channel dice loss")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("loss_plot.png", bbox_inches="tight", dpi=150)
    plt.close() 
    return

def main():
    # Set up
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 30
    model = modules.UNet_2D(in_channels=1, num_categories=5).to(device)
    
    optimiser = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # alters learning rate by factor 0.1 for optimiser when validation dice
    # doesn't increase for 2 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=4)

    best_dice = 0.0
    total_time = 0.0

    train_loss = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    # Train UNet
    print("Training 2D UNet...")
    for e in range(num_epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        # training + validation
        train_loss[e] = train_epoch(model, train_loader, device, optimiser)
        all_channel_dice, prostate_dice, val_loss[e] = validate_epoch(model, val_loader, device)
        scheduler.step(prostate_dice)

        # Report epoch time duration
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        total_time += epoch_time

        # Report per channel dice
        print(f"Epoch {e} validation dice per channel: (elapsed time = {epoch_time:.3f})")
        print(f"    C0 Background/Rectum: {all_channel_dice[0].item()}")
        print(f"    C1  Body/Soft Tissue: {all_channel_dice[1].item()}")
        print(f"    C2              Bone: {all_channel_dice[2].item()}")
        print(f"    C3           Bladder: {all_channel_dice[3].item()}")
        print(f"    C4          Prostate: {all_channel_dice[4].item()}")

        # save model if prostate dice improved
        if prostate_dice > best_dice:
            best_dice = prostate_dice
            torch.save(model.state_dict(), "best_2DUnet.pth")
            print(f"New best model saved: prostate DSC = {prostate_dice:.4f}, epoch = {e}")

    # Report elapsed time and peak VRAM for training + validation
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_gb = peak_vram_bytes / (1024.0 * 1024.0 * 1024.0)
    print(f"Average Training and Validation Time = {(total_time/num_epochs):.3f}s per epoch")
    print(f"Peak VRAM during Training and Validation = {peak_vram_gb:.4f} GB")

    plot_loss_curves(train_loss, val_loss, num_epochs)
    return

if __name__ == "__main__":
    main()