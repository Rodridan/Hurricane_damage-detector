#!/usr/bin/env python
# coding: utf-8


"""
Hurricane Damage Detector - Model Training and Evaluation Script
Author: Daniel Rodriguez Gutierrez
For: Constructor Academy 
"""

import os
from typing import Any, Dict, Sequence, Tuple, Union, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import layers

from loguru import logger


# Optional, for GradCAM
try:
    from tf_keras_vis.gradcam import Gradcam
except ImportError:
    Gradcam = None

# Constants
IMG_DIMS = (128, 128)
BATCH_SIZE = 32
IMG_SHAPE = IMG_DIMS + (3,)
CLASSES = ['no_damage', 'damage']

print('TensorFlow Version:', tf.__version__)

#--------------------------------------------------------------
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
    """
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
def eval_model_on_test(
    model: tf.keras.Model,
    test_dir: str = TEST_DIR,
    img_dims: Tuple[int, int] = IMG_DIMS,
    batch_size: int = 128,
    class_names: Sequence[str] = CLASSES
) -> Tuple[tf.data.Dataset, np.ndarray, np.ndarray]:
    """
    Evaluate model on test dataset, return dataset, true labels, and predictions.
    """
    logger.info("Loading test dataset from: {}", test_dir)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        class_names=class_names,
        seed=42,
        image_size=img_dims,
        batch_size=batch_size,
    )
    # Resize images for model input (e.g., 224x224 if using ResNet)
    test_ds = test_ds.map(
        lambda image, label: (tf.image.resize(image, (224, 224)), label)
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

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

#--------------------------------------------------------------
# SAVE OUTPUT IMAGES
#--------------------------------------------------------------

def save_figure(fig, filename_base: str, output_dir: str = "outputs"):
    """
    Saves the given matplotlib figure as both .png and .tiff in the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{filename_base}.png")
    tiff_path = os.path.join(output_dir, f"{filename_base}.tiff")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(tiff_path, dpi=300, bbox_inches='tight')
    logger.success("Saved figure as {} and {}", png_path, tiff_path)

#--------------------------------------------------------------
# DATA AUGMENTATION
#--------------------------------------------------------------

def augment_pipeline(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Applies random augmentations to an image tensor.
    """
    image = tf.image.resize(image, (224, 224))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)
    return image, label
#--------------------------------------------------------------

def apply_augmentation_to_dataset(dataset: tf.data.Dataset, shuffle_buffer: int = 2000) -> tf.data.Dataset:
    """
    Applies the augmentation pipeline to the dataset and shuffles it.
    """
    logger.info("Applying data augmentation to dataset.")
    augmented_ds = (
        dataset
        .map(augment_pipeline, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(shuffle_buffer)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    logger.success("Augmentation pipeline applied.")
    return augmented_ds
#--------------------------------------------------------------

def apply_resize_to_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Resizes images in the dataset to (224, 224) without augmentation.
    """
    logger.info("Resizing images in dataset for validation/test.")
    resized_ds = (
        dataset
        .map(lambda image, label: (tf.image.resize(image, (224, 224)), label), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    logger.success("Resize pipeline applied.")
    return resized_ds    
#--------------------------------------------------------------    

def visualize_augmentations(
    sample_image: tf.Tensor,
    n: int = 5,
    filename_base: str = "augmentation_visualization"
) -> None:
    """
    Visualizes original and n augmented versions of an image.
    Saves plot as both .png and .tiff in the outputs folder.
    """
    logger.info("Visualizing {} augmentations of a sample image.", n)

    def augment_only_image(image):
        image = tf.image.resize(image, (224, 224))
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, tf.uint8)
        return image.numpy().astype(np.uint8)

    augmented_images = [augment_only_image(sample_image) for _ in range(n)]

    fig = plt.figure(figsize=(15, 3))
    plt.subplot(1, n + 1, 1)
    plt.imshow(sample_image.numpy().astype(np.uint8))
    plt.title('Original')
    plt.axis('off')

    for i, aug_img in enumerate(augmented_images):
        plt.subplot(1, n + 1, i + 2)
        plt.imshow(aug_img)
        plt.title(f'Aug {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    save_figure(fig, filename_base)
    plt.show()
    logger.success("Augmentation visualization complete. Saved as outputs/{}.png and .tiff", filename_base)
    
#--------------------------------------------------------------
# MODEL DEFINITION & TRAINING
#--------------------------------------------------------------

def visualize_samples_from_dataset(dataset: tf.data.Dataset, n: int = 8) -> None:
    """
    Visualizes n sample images from a dataset.
    """
    logger.info("Visualizing {} samples from dataset.", n)
    plt.figure(figsize=(12, 6))
    for images, labels in dataset.take(1):
        for i in range(n):
            ax = plt.subplot(2, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    logger.success("Sample visualization complete.")
#--------------------------------------------------------------

def build_resnet_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
    train_base: bool = False
) -> tf.keras.Model:
    """
    Builds and compiles a ResNet50-based binary classifier model.
    Optionally allows the base model to be trainable (fine-tuning).
    """
    logger.info("Building ResNet50-based model. Train base: {}", train_base)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Freeze or unfreeze base model layers
    for layer in base_model.layers:
        layer.trainable = train_base

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    logger.success("Model built and compiled. Trainable parameters: {:,}", model.count_params())
    return model
#--------------------------------------------------------------

def train_model(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 20
) -> tf.keras.callbacks.History:
    """
    Trains the provided model using the specified datasets.
    """
    logger.info("Starting model training for {} epochs.", epochs)
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
    )
    logger.success("Model training complete.")
    return history

#--------------------------------------------------------------
# EVALUATION & VISUALIZATION
#--------------------------------------------------------------

def plot_classification_summary(
    predictions: Sequence[float],
    test_labels: Sequence[int],
    histories: List[Any],
    title: str = "Classification Results",
    threshold: float = 0.5,
    class_names: Sequence[str] = ("no_damage", "damage"),
    phase_labels: Optional[List[str]] = None,
    filename_base: str = "classification_summary"
) -> None:
    """
    Plots confusion matrix, accuracy/loss curves, and prints classification report.
    Saves plot as .png and .tiff in outputs folder.
    """
    logger.info("Plotting classification summary.")
    binary_preds = (np.array(predictions) > threshold).astype(int)
    cm = confusion_matrix(test_labels, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Prepare metrics
    acc, val_acc, loss, val_loss, phase_starts = [], [], [], [], [0]
    for hist in histories:
        acc.extend(hist.history.get('accuracy', []))
        val_acc.extend(hist.history.get('val_accuracy', []))
        loss.extend(hist.history.get('loss', []))
        val_loss.extend(hist.history.get('val_loss', []))
        phase_starts.append(len(acc))
    phase_starts = phase_starts[:-1]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    disp.plot(ax=axs[0], cmap=plt.cm.Blues, colorbar=False)
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')

    axs[1].plot(acc, label='Train Acc', marker='o')
    axs[1].plot(val_acc, label='Val Acc', marker='s')
    for i, start in enumerate(phase_starts[1:], 1):
        axs[1].axvline(start, color='k', linestyle=':', alpha=0.7)
        if phase_labels and i < len(phase_labels):
            axs[1].text(start + 0.5, axs[1].get_ylim()[1] * 0.97, phase_labels[i],
                        rotation=90, va='top', ha='right', fontsize=9, alpha=0.7)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    axs[2].plot(loss, label='Train Loss', marker='o')
    axs[2].plot(val_loss, label='Val Loss', marker='s')
    for i, start in enumerate(phase_starts[1:], 1):
        axs[2].axvline(start, color='k', linestyle=':', alpha=0.7)
        if phase_labels and i < len(phase_labels):
            axs[2].text(start + 0.5, axs[2].get_ylim()[1] * 0.97, phase_labels[i],
                        rotation=90, va='top', ha='right', fontsize=9, alpha=0.7)
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].set_title('Loss')
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, filename_base)
    plt.show()
    logger.info("Classification report:\n{}", classification_report(test_labels, binary_preds, target_names=class_names))
#--------------------------------------------------------------

def normalize_for_display(img: np.ndarray) -> np.ndarray:
    """
    Normalize image array to [0, 1] float for display, or return uint8 as-is.
    """
    img = np.array(img)
    if img.dtype == np.uint8:
        return img
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        return np.clip((img - img_min) / (img_max - img_min), 0, 1)
    return np.zeros_like(img)
#--------------------------------------------------------------

def plot_false_negatives_positives(
    images: np.ndarray,
    test_labels: np.ndarray,
    binary_preds: np.ndarray,
    n_show: int = 8,
    title: str = "False Negatives and False Positives",
    filename_base: str = "false_negatives_positives"
) -> None:
    """
    Plots and saves up to n_show false negatives and positives.
    """
    logger.info("Visualizing false negatives and false positives (up to {}).", n_show)
    false_negatives = np.where((binary_preds == 0) & (test_labels == 1))[0]
    false_positives = np.where((binary_preds == 1) & (test_labels == 0))[0]
    n_fn = min(n_show, len(false_negatives))
    n_fp = min(n_show, len(false_positives))
    n_cols = max(n_fn, n_fp)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

    for i in range(n_fn):
        idx = false_negatives[i]
        axes[0, i].imshow(normalize_for_display(images[idx]))
        axes[0, i].set_title(f"FN\nIdx: {idx}")
        axes[0, i].axis("off")
    for i in range(n_fn, n_cols):
        axes[0, i].axis("off")

    for i in range(n_fp):
        idx = false_positives[i]
        axes[1, i].imshow(normalize_for_display(images[idx]))
        axes[1, i].set_title(f"FP\nIdx: {idx}")
        axes[1, i].axis("off")
    for i in range(n_fp, n_cols):
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("False Negatives", fontsize=14)
    axes[1, 0].set_ylabel("False Positives", fontsize=14)
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, filename_base)
    plt.show()
    logger.success("False positive/negative visualization complete.")
#--------------------------------------------------------------

def extract_images_and_labels(dataset: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unbatches and extracts all images and labels from a dataset into arrays.
    """
    logger.info("Extracting images and labels from dataset.")
    images, labels = [], []
    for img, lab in dataset.unbatch():
        images.append(img.numpy())
        labels.append(lab.numpy())
    logger.success("Extraction complete: {} images.", len(images))
    return np.array(images), np.array(labels)
#--------------------------------------------------------------

def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    """
    import cv2
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img.astype(np.uint8), 1 - alpha, heatmap_color, alpha, 0)
    return overlay
#--------------------------------------------------------------

def select_image_by_type(
    images: np.ndarray, 
    preds: np.ndarray, 
    labels: np.ndarray, 
    indices: np.ndarray, 
    i: int = 0
) -> Tuple[np.ndarray, int, int, int]:
    """
    Selects an image and its metadata by prediction category indices and position.
    """
    if len(indices) == 0:
        raise ValueError("No samples of this type found.")
    idx = indices[i]
    return images[idx], preds[idx], labels[idx], idx
#--------------------------------------------------------------

def plot_gradcam_for_categories(
    filename_base: str = "gradcam_categories",
    images: np.ndarray,
    binary_preds: np.ndarray,
    test_labels: np.ndarray,
    model: tf.keras.Model,
    loss_func=None,
    idx_dict: Optional[Dict[str, int]] = None
) -> None:
    """
    Plots original images and GradCAM overlays for TP, TN, FP, FN categories.
    """
    if Gradcam is None:
        logger.warning("Gradcam not available. Please install tf-keras-vis.")
        return

    logger.info("Plotting GradCAM overlays for each category.")
    false_negatives = np.where((binary_preds == 0) & (test_labels == 1))[0]
    false_positives = np.where((binary_preds == 1) & (test_labels == 0))[0]
    true_positives  = np.where((binary_preds == 1) & (test_labels == 1))[0]
    true_negatives  = np.where((binary_preds == 0) & (test_labels == 0))[0]
    categories = [
        ("True Positive", true_positives),
        ("True Negative", true_negatives),
        ("False Positive", false_positives),
        ("False Negative", false_negatives)
    ]
    gradcam = Gradcam(model)
    if loss_func is None:
        def loss(output): return output[:, 0]
    else:
        loss = loss_func

    plt.figure(figsize=(18, 8))
    for col, (cat_name, indices) in enumerate(categories):
        i = idx_dict.get(cat_name, 0) if idx_dict else 0
        try:
            image, pred, label, idx = select_image_by_type(images, binary_preds, test_labels, indices, i=i)
        except Exception as e:
            logger.warning(f"Skipping {cat_name}: {e}")
            continue
        image_input = np.expand_dims(image, axis=0).astype(np.float32)
        heatmap = gradcam(loss, image_input)

        ax1 = plt.subplot(2, 4, col + 1)
        ax1.set_title(f"{cat_name}\nIdx: {idx}\nPred: {pred}, Label: {label}", fontsize=12)
        ax1.imshow(image.astype("uint8"))
        ax1.axis('off')

        ax2 = plt.subplot(2, 4, col + 5)
        ax2.set_title("GradCAM", fontsize=12)
        overlay = overlay_heatmap(image.astype("uint8"), heatmap[0])
        ax2.imshow(overlay)
        ax2.axis('off')

    plt.tight_layout()
    save_figure(fig, filename_base)
    plt.show()
    logger.success("GradCAM overlays complete.")
#--------------------------------------------------------------

def plot_samples_category(
    images: np.ndarray,
    binary_preds: np.ndarray,
    test_labels: np.ndarray,
    n_show: int = 8
) -> None:
    """
    Plots a grid of sample images from each prediction category: TP, TN, FP, FN.
    """
    logger.info("Visualizing sample images by category ({} per type).", n_show)
    false_negatives = np.where((binary_preds == 0) & (test_labels == 1))[0]
    false_positives = np.where((binary_preds == 1) & (test_labels == 0))[0]
    true_positives  = np.where((binary_preds == 1) & (test_labels == 1))[0]
    true_negatives  = np.where((binary_preds == 0) & (test_labels == 0))[0]

    categories = [
        ('True Positive', true_positives),
        ('True Negative', true_negatives),
        ('False Positive', false_positives),
        ('False Negative', false_negatives)
    ]

    n_rows = len(categories)
    fig, axes = plt.subplots(n_rows, n_show, figsize=(3 * n_show, 3.5 * n_rows))
    if n_show == 1:
        axes = axes.reshape(-1, 1)

    for row, (cat_name, indices) in enumerate(categories):
        show_count = min(n_show, len(indices))
        for col in range(n_show):
            ax = axes[row, col]
            ax.axis('off')
            if col < show_count:
                idx = indices[col]
                ax.imshow(images[idx].astype("uint8"))
                ax.set_title(f"Idx: {idx}", fontsize=9)
        # Add row label
        fig.text(
            0.08,
            1 - (3.5 * row + 0.5) / n_rows,
            cat_name,
            va='center', ha='center',
            fontsize=14, fontweight='bold', rotation=90
        )

    plt.suptitle("Sample Images by Prediction Category", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    plt.show()
    logger.success("Category sample visualization complete.")
    
#--------------------------------------------------------------
# MAIN
#--------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== Hurricane Damage Detector Pipeline Started ===")

    # Step 1: Prepare data
    train_ds, val_ds = prepare_train_and_val_datasets()
    train_ds = apply_augmentation_to_dataset(train_ds)
    val_ds = apply_resize_to_dataset(val_ds)

    # Step 2: Build and train model
    model = build_resnet_model(input_shape=(224, 224, 3), learning_rate=1e-4, train_base=False)
    history = train_model(model, train_ds, val_ds, epochs=20)

    # Step 3: Evaluate on test set
    test_ds, test_labels, predictions = eval_model_on_test(model)
    binary_preds = (predictions > 0.5).astype(int)

    # Step 4: Plot and save classification summary
    plot_classification_summary(
        predictions=predictions,
        test_labels=test_labels,
        histories=[history],
        class_names=CLASSES,
        title="Classification Results",
        filename_base="classification_summary"
    )

    # Step 5: Extract images for further analysis
    images, labels = extract_images_and_labels(test_ds)

    # Step 6: Visualize and save false negatives/positives
    plot_false_negatives_positives(
        images=images,
        test_labels=test_labels,
        binary_preds=binary_preds,
        n_show=8,
        title="Hurricane Damage Misclassifications",
        filename_base="fp_fn_grid"
    )

    # Step 7: GradCAM visualizations (if Gradcam available)
    idx_dict = {
        "True Positive": 5,
        "True Negative": 10,
        "False Positive": 0,
        "False Negative": 0
    }
    plot_gradcam_for_categories(
        images=images,
        binary_preds=binary_preds,
        test_labels=test_labels,
        model=model,
        idx_dict=idx_dict,
        filename_base="gradcam_categories"
    )

    # Step 8: Grid of sample images per category
    plot_samples_category(
        images=images,
        binary_preds=binary_preds,
        test_labels=test_labels,
        n_show=8,
        filename_base="category_samples"
    )

    logger.success("=== Pipeline completed successfully! ===")
