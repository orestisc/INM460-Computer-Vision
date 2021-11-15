import os

# paths
PATH_TO_DATASET_DIR = "dataset"
PATH_TO_CLASSES_TXT = "configs\classes.txt"
VIDEO_PATH = "dataset/video_elon.mp4"

CHECKPOINT_DIR = "checkpoints"
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

CHECKPOINT_PATH = "checkpoints\checkpoint_epoch_40.tar"

# training parameters
IMAGE_SIZE = (100, 100)
BATCH_SIZE = 4
NUM_CLASSES = 7
NUM_EPOCHS = 50
RESUME = True

MODEL_SAVE_EPOCHS = 10

