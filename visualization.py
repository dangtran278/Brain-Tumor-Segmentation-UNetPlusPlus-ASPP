import os
import torch
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.colors import ListedColormap
import numpy as np

# === Paths ===
output_dir = "outputs"
fixed_index = 42
save_path = os.path.join(output_dir, f"merged_idx{fixed_index}.png")

# === Load ground truth + T1 ===
# _, mask = torch.load("data/brats_test.pt")
# img, gt_mask = mask[fixed_index]  # img: (3, H, W), gt_mask: (H, W)
X_test, y_test = torch.load("data/brats_test.pt")
img = X_test[fixed_index]
gt_mask = y_test[fixed_index]

t1 = img[2].numpy()
t1_norm = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-5)
t1_rgb = np.stack([t1_norm] * 3, axis=-1)

gt_mask_np = gt_mask.numpy()
cmap = ListedColormap(['black', 'red', 'green', 'yellow'])
gt_overlay = 0.5 * t1_rgb + 0.5 * cmap(gt_mask_np)[..., :3]

# === Load predicted overlays ===
paths = {
    "unet": f"overlay_predicted_on_T1_unet_idx{fixed_index}.png",
    "unet++": f"overlay_predicted_on_T1_unetpp_idx{fixed_index}.png",
    "runet++aspp": f"overlay_predicted_on_T1_runetpp_idx{fixed_index}.png"
}
pred_images = [imread(os.path.join(output_dir, fname)) for fname in paths.values()]

# === Plot and merge ===
fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows × 3 columns
titles_top = ['T1 Image', 'Ground Truth', '']  # Leave the third cell empty
titles_bottom = ['UNet','UNet++' ,'Enhanced UNet++ ASPP']

# First row
axes[0, 0].imshow(t1_rgb)
axes[0, 0].set_title(titles_top[0], fontsize=16)
axes[0, 0].axis('off')

axes[0, 1].imshow(gt_overlay)
axes[0, 1].set_title(titles_top[1], fontsize=16)
axes[0, 1].axis('off')

axes[0, 2].axis('off')  # Empty cell

# Second row
for i, (img, title) in enumerate(zip(pred_images, titles_bottom)):
    axes[1, i].imshow(img)
    axes[1, i].set_title(title, fontsize=16)
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig(save_path)
print(f"Saved merged image to {save_path}")