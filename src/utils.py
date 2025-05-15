import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap


# Visualize segmentation overlays
def visualize_segmentation_overlay(
    model, dataloader, num_samples=4, device=torch.device("cuda")
):
    model.eval()
    _, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            batch_size = images.size(0)
            for b in range(batch_size):
                if count >= num_samples:
                    break

                image = images[b].to(device).unsqueeze(0)  # [1, 3, H, W]
                mask = masks[b].to(device)  # [H, W]

                output = model(image)  # [1, C, H, W]
                pred_mask = (
                    torch.argmax(torch.softmax(output, dim=1), dim=1)[0].cpu().numpy()
                )
                mask_np = mask.cpu().numpy()
                flair_np = image[0, 0].cpu().numpy()

                flair_np = (flair_np - flair_np.min()) / (
                    flair_np.max() - flair_np.min() + 1e-5
                )
                flair_rgb = np.stack([flair_np] * 3, axis=-1)

                cmap = ListedColormap(["black", "red", "green", "yellow"])
                gt_overlay = 0.5 * flair_rgb + 0.5 * cmap(mask_np)[..., :3]
                pred_overlay = 0.5 * flair_rgb + 0.5 * cmap(pred_mask)[..., :3]

                # Plotting
                axes[count, 0].imshow(flair_np, cmap="gray")
                axes[count, 0].set_title("FLAIR Image")
                axes[count, 0].axis("off")

                axes[count, 1].imshow(gt_overlay)
                axes[count, 1].set_title("Ground Truth Overlay")
                axes[count, 1].axis("off")

                axes[count, 2].imshow(pred_overlay)
                axes[count, 2].set_title("Predicted Overlay")
                axes[count, 2].axis("off")

                count += 1
            if count >= num_samples:
                break

    plt.tight_layout()
    plt.show()
