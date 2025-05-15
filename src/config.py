# Parameters
DATA_DIR = "./data/"
DATASET = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
TRAIN_SET = "brats_train.pt"
VAL_SET = "brats_val.pt"
TEST_SET = "brats_test.pt"

CHECKPOINT_DIR = "./checkpoints/"
CHECKPOINT = "unet++aspp_dce5e-1_invfreqw_adam1e-3_batch32_epoch50.pth"

MODEL = "unet++aspp"  # "unet", "unet++"

IMG_SIZE = 128
VOLUME_START_AT = 60
VOLUME_SLICES = 75

INPUT_CHANNELS = 3
NUM_CLASSES = 4
CLASS_WEIGHTS = None  # Use inverse class frequency
DICE_WEIGHT = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50
