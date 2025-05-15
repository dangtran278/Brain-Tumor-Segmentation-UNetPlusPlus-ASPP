import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.config as config
from src.dataset import BraTSDataset
from src.loss import DiceCELoss
from src.metrics import IoU
from src.models.unet import UNet
from src.models.unetpp import NestedUNet
from src.models.unetpp_aspp import NestedUNetASPP

# Parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_CHANNELS = config.INPUT_CHANNELS
NUM_CLASSES = config.NUM_CLASSES
CLASS_WEIGHTS = config.CLASS_WEIGHTS
DICE_WEIGHT = config.DICE_WEIGHT
LEARNING_RATE = config.LEARNING_RATE
BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCHS = config.NUM_EPOCHS

TRAIN_PATH = os.path.join(config.DATA_DIR, config.TRAIN_SET)
TEST_PATH = os.path.join(config.DATA_DIR, config.TEST_SET)
SAVE_PATH = f"checkpoints/{config.MODEL}_dce{DICE_WEIGHT:.0e}_invfreqw_adam{LEARNING_RATE:.0e}_batch{BATCH_SIZE}_epoch{NUM_EPOCHS}.pth".replace(
    "+0", ""
).replace(
    "-0", "-"
)


def compute_class_weights(y_train, num_classes):
    y_flat = y_train.view(-1)
    counts = torch.bincount(y_flat, minlength=num_classes).float()
    total = counts.sum()
    weights = total / counts
    return weights / weights.sum()


def get_model(model_name):
    model_dict = {
        "unet": UNet,
        "unet++": NestedUNet,
        "unet++aspp": NestedUNetASPP,
    }
    if model_name not in model_dict:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_dict[model_name]


def train(model, train_loader, num_epochs, optimizer, criterion, device, save_path):
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_iou = 0.0
            num_samples = 0

            for _, batch in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{num_epochs}",
            ):
                inputs, labels = batch
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)

                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                with torch.no_grad():  # Disable gradient calculation
                    iou = IoU(out, labels)
                    total_iou += iou * inputs.size(0)
                    num_samples += inputs.size(0)

                total_loss += loss.item() * inputs.size(0)

            average_loss = total_loss / num_samples
            average_iou = total_iou / num_samples
            print(
                "  Average Loss: {:.4f}, Average IoU: {:.4f}".format(
                    average_loss, average_iou
                )
            )

        torch.save(model, save_path)

    except Exception as e:
        print(e)
        torch.save(model, save_path)


def test(
    model, dataloader, device=DEVICE, num_classes=4, eps=torch.finfo(torch.float32).eps
):
    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    num_samples = 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs = imgs.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            outputs = model(imgs)  # (B, C, H, W)
            pred_masks = torch.argmax(outputs, dim=1)  # (B, H, W)

            dice_sum = 0.0
            iou_sum = 0.0
            for c in range(num_classes):
                pred_cls = (pred_masks == c).float()
                true_cls = (masks == c).float()

                intersection = torch.sum(pred_cls * true_cls)
                union = torch.sum((pred_cls + true_cls) > 0)
                iou = (intersection + eps) / (union + eps)
                dice = (2.0 * intersection + eps) / (
                    torch.sum(pred_cls) + torch.sum(true_cls) + eps
                )

                dice_sum += dice.item()
                iou_sum += iou.item()

            total_dice += dice_sum / num_classes
            total_iou += iou_sum / num_classes
            num_samples += 1

    return {"IoU": total_iou / num_samples, "Dice": total_dice / num_samples}


def main():
    X_train, y_train = torch.load(TRAIN_PATH)
    X_test, y_test = torch.load(TEST_PATH)
    train_dataset = BraTSDataset(X_train, y_train)
    test_dataset = BraTSDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_cls = get_model(config.MODEL)
    model = model_cls(NUM_CLASSES, INPUT_CHANNELS)
    criterion = DiceCELoss(
        dice_weight=DICE_WEIGHT,
        class_weights=CLASS_WEIGHTS
        if CLASS_WEIGHTS is not None
        else compute_class_weights(y_train, NUM_CLASSES).to(DEVICE),
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model = model.to(DEVICE)
    train(model, train_loader, NUM_EPOCHS, optimizer, criterion, DEVICE, SAVE_PATH)

    metrics = test(model, test_loader, device=DEVICE)
    print(f"Test IoU: {metrics['IoU']:.4f}")
    print(f"Test Dice: {metrics['Dice']:.4f}")
