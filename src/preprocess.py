import os
import random

import cv2
import nibabel as nib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config

# Dataset: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation.
# Download and unzip in ./data folder.


# ----------- Configuration -----------
IMG_SIZE = config.IMG_SIZE
VOLUME_START_AT = config.VOLUME_START_AT
VOLUME_SLICES = config.VOLUME_SLICES
SAVE_DIR = config.DATA_DIR
DATA_PATH = os.path.join(SAVE_DIR, config.DATASET)
TRAIN_SET = config.TRAIN_SET
VAL_SET = config.VAL_SET
TEST_SET = config.TEST_SET


def get_foreground_crop_coords(img, margin=5):
    """Return crop coordinates [y1, y2, x1, x2] that trim black borders."""
    if np.max(img) == 0:
        return 0, img.shape[0], 0, img.shape[1]

    img_bin = img > 0
    coords = cv2.findNonZero(img_bin.astype(np.uint8))
    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, img.shape[1])
    y2 = min(y + h + margin, img.shape[0])
    return y1, y2, x1, x2


def pad_to_square(img):
    """Pads a 2D array to make it square."""
    h, w = img.shape
    if h == w:
        return img
    diff = abs(h - w)
    if h > w:
        pad_left = diff // 2
        pad_right = diff - pad_left
        padding = ((0, 0), (pad_left, pad_right))
    else:
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        padding = ((pad_top, pad_bottom), (0, 0))
    return np.pad(img, padding, mode="constant", constant_values=0)


def get_largest_tumor_slice(seg, axis):
    """Return index of slice with the largest tumor area along the given axis."""
    max_area = -1
    best_idx = 0
    for i in range(seg.shape[axis]):
        if axis == 0:
            slice_ = seg[i, :, :]
        elif axis == 1:
            slice_ = seg[:, i, :]
        else:  # axis == 2
            slice_ = seg[:, :, i]
        area = np.sum(slice_ > 0)
        if area > max_area:
            max_area = area
            best_idx = i
    return best_idx


def process_samples(sample_list, use_cropping=True):
    X_all = []
    y_all = []

    for _, pid in enumerate(tqdm(sample_list)):
        prefix = os.path.join(DATA_PATH, pid, pid)
        flair = nib.load(prefix + "_flair.nii").get_fdata()
        t1ce = nib.load(prefix + "_t1ce.nii").get_fdata()
        t2 = nib.load(prefix + "_t2.nii").get_fdata()
        seg = nib.load(prefix + "_seg.nii").get_fdata()

        axial_idx = get_largest_tumor_slice(seg, axis=2)
        coronal_idx = get_largest_tumor_slice(seg, axis=1)
        sagittal_idx = get_largest_tumor_slice(seg, axis=0)

        views = [
            flair[:, :, axial_idx],
            flair[:, coronal_idx, :],
            flair[sagittal_idx, :, :],
        ]
        t1ce_views = [
            t1ce[:, :, axial_idx],
            t1ce[:, coronal_idx, :],
            t1ce[sagittal_idx, :, :],
        ]
        t2_views = [t2[:, :, axial_idx], t2[:, coronal_idx, :], t2[sagittal_idx, :, :]]
        seg_views = [
            seg[:, :, axial_idx],
            seg[:, coronal_idx, :],
            seg[sagittal_idx, :, :],
        ]

        for v in range(3):
            # --- Crop
            if use_cropping:
                y1, y2, x1, x2 = get_foreground_crop_coords(views[v])
                flair_crop = views[v][y1:y2, x1:x2]
                t1ce_crop = t1ce_views[v][y1:y2, x1:x2]
                t2_crop = t2_views[v][y1:y2, x1:x2]
                seg_crop = seg_views[v][y1:y2, x1:x2]
            else:
                flair_crop = views[v]
                t1ce_crop = t1ce_views[v]
                t2_crop = t2_views[v]
                seg_crop = seg_views[v]

            # --- Pad to square
            flair_crop = pad_to_square(flair_crop)
            t1ce_crop = pad_to_square(t1ce_crop)
            t2_crop = pad_to_square(t2_crop)
            seg_crop = pad_to_square(seg_crop)

            # --- Resize
            flair_resized = cv2.resize(flair_crop, (IMG_SIZE, IMG_SIZE))
            t1ce_resized = cv2.resize(t1ce_crop, (IMG_SIZE, IMG_SIZE))
            t2_resized = cv2.resize(t2_crop, (IMG_SIZE, IMG_SIZE))
            seg_resized = cv2.resize(
                seg_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
            )

            seg_resized[seg_resized == 4] = 3

            x = np.stack([flair_resized, t1ce_resized, t2_resized], axis=0)
            x_max = np.max(x)
            x = x / x_max if x_max > 0 else x

            X_all.append(torch.tensor(x, dtype=torch.float32))
            y_all.append(torch.tensor(seg_resized, dtype=torch.long))

    return torch.stack(X_all), torch.stack(y_all)


def main():
    # ----------- Get patient IDs -----------
    samples = [
        s
        for s in os.listdir(DATA_PATH)
        if "BraTS20" in s and s != "BraTS20_Training_355"
    ]
    samples_train, samples_val = train_test_split(
        samples, test_size=0.2, random_state=42
    )
    samples_train, samples_test = train_test_split(
        samples_train, test_size=0.15, random_state=42
    )

    # ----------- Process and save datasets -----------
    random.shuffle(samples_train)
    X_train, y_train = process_samples(samples_train, use_cropping=False)

    # Just to check the number of slices
    print("Total patients in training:", len(samples_train))
    print("Expected number of slices:", len(samples_train) * 3)
    print("Actual number of slices:", len(X_train))
    assert (
        len(X_train) == len(samples_train) * 3
    ), "Mismatch: more or fewer slices than expected!"

    X_val, y_val = process_samples(samples_val, use_cropping=False)
    X_test, y_test = process_samples(samples_test, use_cropping=False)

    torch.save((X_train, y_train), os.path.join(SAVE_DIR, TRAIN_SET))
    torch.save((X_val, y_val), os.path.join(SAVE_DIR, VAL_SET))
    torch.save((X_test, y_test), os.path.join(SAVE_DIR, TEST_SET))
