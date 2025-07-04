import os, gdown, zipfile, tensorflow as tf
from loguru import logger
from riskscope.config import BATCH_SIZE, CLASSES, IMG_DIMS,EVAL_BATCH_SIZE
from typing import Tuple, Sequence
import numpy as np
from tqdm import tqdm
#--------------------------------------------------------------
#DATA DOWBLOADING & EXTRACTION
#--------------------------------------------------------------
def get_test_dataset(test_dir, img_dims=(224, 224), class_names=None):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        class_names=class_names,
        seed=42,
        image_size=img_dims,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False
    )
    test_ds = test_ds.prefetch(buffer_size=1)
    return test_ds

def download_and_extract_data(
    gdrive_id: str,
    data_dir: str = "data",
    zip_filename: str = "hurricane_detector.zip",
    extracted_folder_name: str = "train_hurricane",
    overwrite: bool = False
) -> str:
    """
    Download and extract data from Google Drive if not already present.

    Parameters:
        gdrive_id: Google Drive file ID
        data_dir: Where to store/extract data
        zip_filename: What to name the downloaded zip
        extracted_folder_name: Folder inside zip to use as the dataset root
        overwrite: If True, re-download and extract even if data exists

    Returns:
        Path to extracted data directory
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, zip_filename)
    extract_path = os.path.join(data_dir, extracted_folder_name)
    
    if os.path.exists(extract_path) and not overwrite:
        logger.info(f"Data already exists at {extract_path}, skipping download.")
        return extract_path

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    logger.info(f"Downloading data from {url} ...")
    gdown.download(url, zip_path, quiet=False)
    logger.success(f"Downloaded to {zip_path}")

    logger.info(f"Extracting {zip_path} to {data_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    logger.success(f"Extracted dataset to {extract_path}")
    
    return extract_path

# Data directories (adjust as needed)
TRAIN_DIR = "./train_hurricane"
TEST_DIR = "./test_hurricane"

#--------------------------------------------------------------
#PREPROCESSING AND TRAIN/VAL SPLIT
#--------------------------------------------------------------
def prepare_train_and_val_datasets(
    train_dir: str = TRAIN_DIR,
    img_dims: Tuple[int, int] = IMG_DIMS,
    batch_size: int = BATCH_SIZE,
    validation_split: float = 0.2,
    seed: int = 42,
    class_names: Sequence[str] = CLASSES
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Prepares the TensorFlow datasets for training and validation.

    Args:
        train_dir: Directory containing class folders with images.
        img_dims: Tuple specifying the image size (height, width).
        batch_size: Batch size for data loading.
        validation_split: Fraction of data to use for validation.
        seed: Random seed for reproducibility.
        class_names: List of class names corresponding to subdirectories.

    Returns:
        train_ds: Training dataset.
        val_ds: Validation dataset.
    """
    if not os.path.exists(train_dir):
        logger.error(
            f"Training directory '{train_dir}' does not exist!\n"
            f"Please ensure your data is extracted to this path.\n"
            f"Expected structure: {train_dir}/<class_name>/image1.jpg"
        )
        raise FileNotFoundError(f"Training directory '{train_dir}' does not exist.")

    logger.info("Preparing training and validation datasets from directory: {}", train_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="training",
        class_names=class_names,
        seed=seed,
        image_size=img_dims,
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset="validation",
        class_names=class_names,
        seed=seed,
        image_size=img_dims,
        batch_size=batch_size,
    )

    logger.success("Datasets prepared: {} training batches, {} validation batches",
                   len(train_ds), len(val_ds))

    train_ds = train_ds.prefetch(buffer_size=25)
    val_ds = val_ds.prefetch(buffer_size=25)

    return train_ds, val_ds

#--------------------------------------------------------------
def get_test_dataset(
    test_dir: str,
    img_dims: Tuple[int, int],
    batch_size: int,
    class_names: Sequence[str]
) -> tf.data.Dataset:
    """
    Loads test dataset for evaluation and extraction, using safe batch size and prefetch.
    """
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        class_names=class_names,
        seed=42,
        image_size=img_dims,
        batch_size=batch_size,
        shuffle=False
    )
    # Always resize to model input (e.g., 224x224)
    test_ds = test_ds.map(
        lambda image, label: (tf.image.resize(image, (224, 224)), label)
    ).prefetch(buffer_size=1)
    return test_ds

def eval_model_on_test(
    model: tf.keras.Model,
    test_dir: str = TEST_DIR,
    img_dims: Tuple[int, int] = IMG_DIMS,
    batch_size: int = 16,   # Use small batch size for evaluation
    class_names: Sequence[str] = CLASSES
) -> Tuple[tf.data.Dataset, np.ndarray, np.ndarray]:
    """
    Evaluate model on test dataset, return dataset, true labels, and predictions.
    """
    logger.info("Loading test dataset from: {}", test_dir)
    test_ds = get_test_dataset(test_dir, img_dims, batch_size, class_names)

    test_labels = []
    predictions = []

    logger.info("Predicting on test data...")
    for imgs, labels in tqdm(test_ds, desc='Predicting on Test Data'):
        batch_preds = model.predict(imgs)
        predictions.extend(batch_preds)
        test_labels.extend(labels.numpy())

    predictions = np.array(predictions).ravel()
    test_labels = np.array(test_labels)

    logger.success("Test predictions completed: {} samples", len(test_labels))
    return test_ds, test_labels, predictions