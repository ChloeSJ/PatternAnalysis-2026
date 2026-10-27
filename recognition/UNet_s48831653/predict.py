import torch
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import modules
import dataset
import train

def save_image(img, gt_masks, pred_masks, i):
    """
    Save figure with original MRI, ground truth segmentation, predicted 
    segmentation and the corresponding error map.
    """
    # assign colours to channels
    label_colours = ["black", "green", "brown", "cyan", "red"]
    label_cmap = ListedColormap(label_colours)

    # Retrieve map of the segmented values
    gt_map = np.argmax(gt_masks, axis=0)
    pred_map = np.argmax(pred_masks, axis=0)

    # Construct RGB error map for prostate segmentation
    gt_prostate = gt_masks[4]
    pred_prostate = pred_masks[4]
    error_map = np.zeros((*gt_map.shape, 3))
    error_map[(pred_prostate == 1) & (gt_prostate == 0)]  = [1, 0, 0] # False positive (Red)
    error_map[(pred_prostate == 0) & (gt_prostate == 1)]  = [0, 0, 1] # False negative (Blue)
    error_map[(pred_prostate == 1) & (gt_prostate == 1)] = [0, 1, 0] # True positive (Green)

    # Construct image
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    # original MRI slice
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Batch {i:03d}: Input MRI Slice")
    axes[0].axis("off")

    # Ground Truth Segmentation Map
    axes[1].imshow(gt_map, cmap=label_cmap, vmin=0, vmax=4)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    #  Model Predicted Segmentation Map
    axes[2].imshow(pred_map, cmap=label_cmap, vmin=0, vmax=4)
    axes[2].set_title(f"Predicted Mask ({i:03d})")
    axes[2].axis("off")

    # Error overlay
    axes[3].imshow(img, cmap="gray")
    axes[3].imshow(error_map, alpha=0.7)
    axes[3].set_title("Error map")
    axes[3].axis("off")

    # Save image
    plt.tight_layout()
    save_path = os.path.join("prediction_images", f"slice_{i:03}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    return

def test(model, dataloader, device):
    model.eval()
    total_intersection = torch.zeros(5, device=device)
    total_pixels = torch.zeros(5, device=device)

    # set up path name for prediction images
    os.makedirs(f"prediction_images", exist_ok=True)
    i = 0 #image index

    with torch.no_grad():
        for images, masks in dataloader:
            # testing
            images, masks = images.to(device), masks.to(device)
            probabilities = model(images)

            intersection, total = train.hard_dice_loss(probabilities, masks)
            total_intersection += intersection.sum(dim=0)
            total_pixels += total.sum(dim=0)
            preds = (probabilities >= 0.5).float()

            # Save each image the current batch
            for j in range(images.size(0)):
                # Retrieve relevant maps
                img = images[j, 0].cpu().numpy() if images.ndim == 4 else images[j].cpu().numpy()
                gt_masks = masks[j].cpu().numpy()
                pred_masks = preds[j].cpu().numpy()
                save_image(img, gt_masks, pred_masks, i)
                i += 1

    eps = 1e-6 # eps for gradient stablisation
    all_channel_dice = (2.0 * total_intersection + eps) / (total_pixels + eps)
    prostate_dice = all_channel_dice[4].item()

    return all_channel_dice, prostate_dice

def main():
    # Set up
    print("Starting evaluation of test set:")
    _, _, test_loader = dataset.get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = modules.UNet_2D(in_channels=1, num_categories=5).to(device)

    # start timing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # predict + save test cases
    model.load_state_dict(torch.load("best_2DUnet.pth"))
    avg_dice, prostate_dice = test(model, test_loader, device)

    # report testing duration, peak VRAM, and dice
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_bytes = torch.cuda.max_memory_allocated(device)
        peak_vram_gb = peak_vram_bytes / (1024.0 * 1024.0 * 1024.0)

    end_time = time.perf_counter()
    print(f"Average Testing Time = {(end_time - start_time)/len(test_loader.dataset)} per MRI slice")
    print(f"Peak VRAM during Testing = {peak_vram_gb:.4f} GB")
    print(f"Test set prostate dice = {prostate_dice:.4f}")
    return

if __name__ == "__main__":
    main()