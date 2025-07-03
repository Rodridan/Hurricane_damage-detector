import os
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Sequence, Any, List, Optional, Dict, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

try:
    from tf_keras_vis.gradcam import Gradcam
except ImportError:
    Gradcam = None

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
# EVALUATION & VISUALIZATION
#--------------------------------------------------------------

def visualize_samples_from_dataset(
    dataset: tf.data.Dataset,
    n: int = 8,
    filename_base: str = "sample_images"
) -> None:
    """
    Visualizes n sample images from a dataset and saves as PNG and TIFF.
    """
    logger.info("Visualizing {} samples from dataset.", n)
    fig = plt.figure(figsize=(12, 6))
    for images, labels in dataset.take(1):
        for i in range(n):
            ax = plt.subplot(2, 4, i + 1)
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.set_title(int(labels[i]))
            ax.axis("off")
    plt.tight_layout()
    save_figure(fig, filename_base)  # Save as PNG and TIFF in outputs/
    plt.show()
    logger.success("Sample visualization complete. Saved as outputs/{}.png and .tiff", filename_base)
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
    images: np.ndarray,
    binary_preds: np.ndarray,
    test_labels: np.ndarray,
    model: tf.keras.Model,
    loss_func=None,
    idx_dict: Optional[Dict[str, int]] = None,
    filename_base: str = "gradcam_categories"
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
