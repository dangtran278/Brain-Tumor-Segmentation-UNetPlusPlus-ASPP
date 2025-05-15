import os

import torch
from torch.utils.data import DataLoader

import src.config as config
from src.dataset import BraTSDataset
from src.models.unet import UNet
from src.models.unetpp import NestedUNet
from src.models.unetpp_aspp import NestedUNetASPP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_CHANNELS = config.INPUT_CHANNELS
NUM_CLASSES = config.NUM_CLASSES
BATCH_SIZE = config.BATCH_SIZE
TEST_PATH = os.path.join(config.DATA_DIR, config.TEST_SET)

SAVE_PATH = f"checkpoints/{config.MODEL}_dce{config.DICE_WEIGHT:.0e}_invfreqw_adam{config.LEARNING_RATE:.0e}_batch{BATCH_SIZE}_epoch{config.NUM_EPOCHS}.pth".replace(
    "+0", ""
).replace(
    "-0", "-"
)


def get_model(model_name):
    model_dict = {
        "unet": UNet,
        "unet++": NestedUNet,
        "unet++aspp": NestedUNetASPP,
    }
    if model_name not in model_dict:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_dict[model_name]


def load_model(checkpoint_path, model_name):
    model_cls = get_model(model_name)
    model = model_cls(NUM_CLASSES, INPUT_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    return model


def inference(model, dataloader):
    all_preds = []
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(DEVICE, dtype=torch.float32)
            outputs = model(imgs)  # (B, C, H, W)
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)
            all_preds.append(preds.cpu())
    return torch.cat(all_preds, dim=0)


def main():
    X_test, y_test = torch.load(TEST_PATH)
    test_dataset = BraTSDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model(SAVE_PATH, config.MODEL)

    predictions = inference(model, test_loader)
    torch.save(predictions, "predictions.pt")


if __name__ == "__main__":
    main()
