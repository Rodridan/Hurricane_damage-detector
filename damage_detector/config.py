IMG_DIMS = (128, 128)
BATCH_SIZE = 8
CLASSES = ['no_damage', 'damage']
GDRIVE_ID = "1pByxsenTnJGBKnKhLTXBqbUN_Kbm7PNK"
DATA_DIR = "data"
TRAIN_SUBDIR = "train_hurricane"
TEST_SUBDIR = "test_hurricane"

USE_PRETRAINED = True  # Set to False to train a new model from scratch
PRETRAINED_MODEL_PATH = "pretrained_models/hurricane_resnet50_224x224_b32_e20_ph2_stdaug_20250703.keras"
