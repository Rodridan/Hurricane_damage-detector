#!/usr/bin/env python
# coding: utf-8

# # Hurricane Damage Detector

# ## Get and Load Dataset

# In[1]:


get_ipython().system('pip install --upgrade --no-cache-dir gdown')


# In[2]:


get_ipython().system('gdown --id 1pByxsenTnJGBKnKhLTXBqbUN_Kbm7PNK')


# In[3]:


get_ipython().system('unzip -q hurricane_detector.zip')


# In[4]:


ls -l


# In[5]:


get_ipython().system('sudo apt-get install tree')


# In[6]:


get_ipython().system('tree --dirsfirst --filelimit 2 ./train_hurricane/')


# In[7]:


get_ipython().system('tree --dirsfirst --filelimit 2 ./test_hurricane/')


# ## Load Dependencies

# In[8]:


# === Standard Library Imports ===
from typing import Any, Dict, Sequence, Tuple, Union

# === Third-Party Imports ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

# === Scikit-learn Imports ===
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# === TensorFlow/Keras Imports ===
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization
)
from tensorflow.keras import layers

# === tf-keras-vis Import ===
from tf_keras_vis.gradcam import Gradcam

# === Version Info ===
print('TensorFlow Version:', tf.__version__)


# ## Utility functions to create dataset generators

# In[9]:


IMG_DIMS = (128, 128)
BATCH_SIZE = 32
IMG_SHAPE = IMG_DIMS + (3,)
classes = ['no_damage', 'damage']

# call this function before running any model to get data into train and validation splits
# data is loaded as a TF dataset in a memory efficient format
def prepare_train_and_val_datasets():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "./train_hurricane",
        validation_split=0.2,
        subset="training",
        class_names=['no_damage', 'damage'],
        seed=42,
        image_size=IMG_DIMS,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "./train_hurricane",
        validation_split=0.2,
        subset="validation",
        class_names=['no_damage', 'damage'],
        seed=42,
        image_size=IMG_DIMS,
        batch_size=BATCH_SIZE,
    )

    train_ds = train_ds.prefetch(buffer_size=25)
    val_ds = val_ds.prefetch(buffer_size=25)

    return train_ds, val_ds

# call this function on any trained model to get prediction labels on the test data
# this loads the test dataset from the test directory as a test dataset
# iterates through the above dataset and returns the true labels as well as the predicted labels
def eval_model_on_test(model):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "./test_hurricane",
        class_names=['no_damage', 'damage'],
        seed=42,
        image_size=IMG_DIMS,
        batch_size=128,
    )
    test_ds = test_ds.map(lambda image, label: (tf.image.resize(image, (224, 224)), label)).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_labels = []
    predictions = []

    for imgs, labels in tqdm(test_ds.take(100),
                             desc='Predicting on Test Data'):
        batch_preds = model.predict(imgs)
        predictions.extend(batch_preds)
        test_labels.extend(labels)

    predictions = np.array(predictions)
    predictions = predictions.ravel()
    test_labels = np.array(test_labels)

    return test_ds,test_labels, predictions


# In[10]:


train_ds, val_ds = prepare_train_and_val_datasets()
train_ds.take(1)


# # Data augmentation:

# In[11]:


def augment_pipeline(image, label):
    image = tf.image.resize(image, (224, 224))  # Resize to (224, 224)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0, 255)          # Clip to valid range
    image = tf.cast(image, tf.uint8)                 # Cast if necessary
    return image, label

train_ds = train_ds.map(augment_pipeline).shuffle(2000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda image, label: (tf.image.resize(image, (224, 224)), label)).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[12]:


# Number of augmentations to visualize
N = 5

# Retrieve a sample image and label from train_ds (take(1) returns a dataset of one element)
for sample_img, sample_label in train_ds.unbatch().take(1):
    img = sample_img
    break

# Function to apply augmentation pipeline (for visualization)
def augment_only_image(image):
    image = tf.image.resize(image, (224, 224))  # Resize to (224, 224)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0, 255)          # Clip to valid range
    image = tf.cast(image, tf.uint8)                 # Cast if necessary
    return image.numpy().astype(np.uint8)

# Generate N augmented versions
augmented_images = [augment_only_image(img) for _ in range(N)]

# Plot original + N augmentations
plt.figure(figsize=(15, 3))
plt.subplot(1, N + 1, 1)
plt.imshow(img.numpy().astype(np.uint8))
plt.title('Original')
plt.axis('off')

for i, aug_img in enumerate(augmented_images):
    plt.subplot(1, N + 1, i + 2)
    plt.imshow(aug_img)
    plt.title(f'Aug {i + 1}')
    plt.axis('off')

plt.tight_layout()
plt.show()


# # Data visualization:

# In[13]:


plt.figure(figsize=(12, 6))

for images, labels in train_ds.take(1):
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
plt.tight_layout()


# # Base model:
# resnet50
# 

# In[14]:


# Load base model (without top)
base_model = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)


# In[15]:


base_model.summary()


# In[16]:


for layer in base_model.layers[:]:
  if layer.trainable == True:
    print(layer.name, layer.trainable)


# In[17]:


for layer in base_model.layers[:]:
    layer.trainable = False

for layer in base_model.layers[:]:
  if layer.trainable == True:
    print(layer.name, layer.trainable)


# In[18]:


# Add custom layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)

# Construct the model
new_model = Model(inputs=base_model.input, outputs=output)


# In[19]:


new_model.summary()


# In[20]:


new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[21]:


history = new_model.fit(
    train_ds, epochs=20,
    validation_data=val_ds,
)

test_ds, test_labels, predictions = eval_model_on_test(new_model)


# In[22]:


binary_preds = (predictions > 0.5).astype(int)
binary_preds.shape, test_labels.shape


# In[23]:


from typing import Sequence, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def plot_classification_summary_multifit(
    predictions: Sequence[float],
    test_labels: Sequence[int],
    histories: List[Any],
    title: str = "Classification Results",
    threshold: float = 0.5,
    class_names: Sequence[str] = ("no_damage", "damage"),
    phase_labels: Optional[List[str]] = None
) -> None:
    """
    Plots confusion matrix, concatenated training/validation accuracy and loss curves for multiple fits,
    and prints classification metrics for binary classification.

    Args:
        predictions (Sequence[float]): Model predictions (probabilities or logits).
        test_labels (Sequence[int]): True labels.
        histories (List[Any]): List of Keras History objects (from model.fit()).
        title (str): Plot title.
        threshold (float): Threshold to binarize predictions.
        class_names (Sequence[str]): Class names.
        phase_labels (Optional[List[str]]): Optional labels for each training phase.
    """
    # Prepare binary predictions
    binary_preds = (np.array(predictions) > threshold).astype(int)
    cm = confusion_matrix(test_labels, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Prepare concatenated metrics
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    phase_starts = [0]
    for hist in histories:
        acc.extend(hist.history.get('accuracy', []))
        val_acc.extend(hist.history.get('val_accuracy', []))
        loss.extend(hist.history.get('loss', []))
        val_loss.extend(hist.history.get('val_loss', []))
        phase_starts.append(len(acc))
    # Remove the final redundant index
    phase_starts = phase_starts[:-1]
    n_epochs = len(acc)

    # Multi-plot
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))

    # 1. Confusion Matrix
    disp.plot(ax=axs[0], cmap=plt.cm.Blues, colorbar=False)
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xlabel('Predicted Label')
    axs[0].set_ylabel('True Label')

    # 2. Accuracy Plot
    axs[1].plot(acc, label='Training Accuracy', marker='o')
    axs[1].plot(val_acc, label='Validation Accuracy', marker='s')
    for i, start in enumerate(phase_starts[1:], 1):
        axs[1].axvline(start, color='k', linestyle=':', alpha=0.7)
        if phase_labels and i < len(phase_labels):
            axs[1].text(start + 0.5, axs[1].get_ylim()[1] * 0.97, phase_labels[i], rotation=90, va='top', ha='right', fontsize=9, alpha=0.7)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # 3. Loss Plot
    axs[2].plot(loss, label='Training Loss', marker='o')
    axs[2].plot(val_loss, label='Validation Loss', marker='s')
    for i, start in enumerate(phase_starts[1:], 1):
        axs[2].axvline(start, color='k', linestyle=':', alpha=0.7)
        if phase_labels and i < len(phase_labels):
            axs[2].text(start + 0.5, axs[2].get_ylim()[1] * 0.97, phase_labels[i], rotation=90, va='top', ha='right', fontsize=9, alpha=0.7)
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].set_title('Loss')
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Print detailed classification metrics
    print(classification_report(test_labels, binary_preds, target_names=class_names))


# In[24]:


plot_classification_summary_multifit(
    predictions=binary_preds,
    test_labels=test_labels,
    histories=[history],  # List of Keras History objects
    title="Classification Results with Two Training Phases",
    class_names=["no_damage", "damage"],
    phase_labels=["Phase 1", "Phase 2"]
)


# In[25]:


def normalize_for_display(img):
    """
    Normalize any image array to [0, 1] float range for correct matplotlib display.
    Handles both float and integer input types.
    """
    img = np.array(img)
    if img.dtype == np.uint8:
        return img
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        norm_img = (img - img_min) / (img_max - img_min)
        norm_img = np.clip(norm_img, 0, 1)
        return norm_img
    else:
        return np.zeros_like(img)
    plt.imshow(normalize_for_display(images[idx]))


# In[26]:


def plot_false_negatives_positives(
    images: np.ndarray,
    test_labels: np.ndarray,
    binary_preds: np.ndarray,
    n_show: int = 8,
    title: str = "False Negatives and False Positives"
) -> None:
    images = np.array(images)
    test_labels = np.array(test_labels)
    binary_preds = np.array(binary_preds)

    false_negatives = np.where((binary_preds == 0) & (test_labels == 1))[0]
    false_positives = np.where((binary_preds == 1) & (test_labels == 0))[0]

    n_fn = min(n_show, len(false_negatives))
    n_fp = min(n_show, len(false_positives))
    n_cols = max(n_fn, n_fp)
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))

    if n_cols == 1:
        axes = axes.reshape(2, 1)

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
    plt.show()


# In[33]:


# Extract all images and labels for easy indexing
images = []
labels = []
for img, lab in test_ds.unbatch():
    images.append(img.numpy())
    labels.append(lab.numpy())
images = np.array(images)
labels = np.array(labels)

false_negatives = np.where((binary_preds == 0) & (test_labels == 1))[0]
false_positives = np.where((binary_preds == 1) & (test_labels == 0))[0]
true_positives  = np.where((binary_preds == 1) & (test_labels == 1))[0]
true_negatives  = np.where((binary_preds == 0) & (test_labels == 0))[0]

def select_image_by_type(images, preds, labels, indices, i=0):
    idx = indices[i]
    return images[idx], preds[idx], labels[idx], idx

image_number = 10
image, pred, label, idx = select_image_by_type(images, binary_preds, test_labels, true_negatives, i=image_number)
print(f"Selected image idx: {idx}, prediction label: {pred}, true label: {label}")
print(image.shape)


# In[34]:


normalize_for_display(images[idx])
plot_false_negatives_positives(
    images=images,
    test_labels=test_labels,
    binary_preds=binary_preds,
    n_show=8,
    title="Hurricane Damage Misclassifications"
)


# # Refine the model:

# In[35]:


# make all layers trainable
for layer in new_model.layers[:]:
    layer.trainable = True
# make only last layer of resnet50 trainable (except bn)
for layer in new_model.layers[:]:
    if 'conv5' not in layer.name:
        layer.trainable = False
    elif 'bn' in layer.name:
        layer.trainable = False
# Ensure all custom head layers are trainable
for layer in new_model.layers:
    if layer.name not in [l.name for l in base_model.layers]:
        layer.trainable = True

for layer in new_model.layers[:]:
  if layer.trainable == True:
    print(layer.name, layer.trainable)


# In[36]:


new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[37]:


history1 = new_model.fit(
    train_ds, epochs=20,
    validation_data=val_ds,
)

test_ds, test_labels, predictions = eval_model_on_test(new_model)


# In[38]:


plot_classification_summary_multifit(
    predictions=binary_preds,
    test_labels=test_labels,
    histories=[history, history1],
    title="Classification Results",
    class_names=["no_damage", "damage"],
    phase_labels=["Phase 1", "Phase 2"]
)


# In[46]:


# Extract all images and labels for easy indexing
images = []
labels = []
for img, lab in test_ds.unbatch():
    images.append(img.numpy())
    labels.append(lab.numpy())
images = np.array(images)
labels = np.array(labels)

false_negatives = np.where((binary_preds == 0) & (test_labels == 1))[0]
false_positives = np.where((binary_preds == 1) & (test_labels == 0))[0]
true_positives  = np.where((binary_preds == 1) & (test_labels == 1))[0]
true_negatives  = np.where((binary_preds == 0) & (test_labels == 0))[0]

def select_image_by_type(images, preds, labels, indices, i=0):
    idx = indices[i]
    return images[idx], preds[idx], labels[idx], idx

image_number = 10
image, pred, label, idx = select_image_by_type(images, binary_preds, test_labels, true_negatives, i=image_number)
print(f"Selected image idx: {idx}, prediction label: {pred}, true label: {label}")
print(image.shape)


# In[47]:


normalize_for_display(images[idx])
plot_false_negatives_positives(
    images=images,
    test_labels=test_labels,
    binary_preds=binary_preds,
    n_show=8,
    title="Hurricane Damage Misclassifications"
)


# In[48]:


# Extract all images and labels for easy indexing
images = []
labels = []
for img, lab in test_ds.unbatch():
    images.append(img.numpy())
    labels.append(lab.numpy())
images = np.array(images)
labels = np.array(labels)

false_negatives = np.where((binary_preds == 0) & (test_labels == 1))[0]
false_positives = np.where((binary_preds == 1) & (test_labels == 0))[0]
true_positives  = np.where((binary_preds == 1) & (test_labels == 1))[0]
true_negatives  = np.where((binary_preds == 0) & (test_labels == 0))[0]


# In[49]:


image_number = 10
image, pred, label, idx = select_image_by_type(images, binary_preds, test_labels, true_negatives, i=image_number)
print(f"Selected image idx: {idx}, prediction label: {pred}, true label: {label}")
print(image.shape)


# In[50]:


def overlay_heatmap(img, heatmap, alpha=0.4):
    import cv2
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def select_image_by_type(images, preds, labels, indices, i=0):
    if len(indices) == 0:
        raise ValueError("No samples of this type found.")
    idx = indices[i]
    return images[idx], preds[idx], labels[idx], idx

def plot_gradcam_for_categories(
    images, binary_preds, test_labels, model, loss_func=None, idx_dict=None
):
    # Compute indices for each category
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
        # Index selection: if idx_dict provided, use it, otherwise default to 0
        i = idx_dict.get(cat_name, 0) if idx_dict else 0
        try:
            image, pred, label, idx = select_image_by_type(images, binary_preds, test_labels, indices, i=i)
        except Exception as e:
            print(f"Skipping {cat_name}: {e}")
            continue
        image_input = np.expand_dims(image, axis=0).astype(np.float32)
        heatmap = gradcam(loss, image_input)

        # Original image (top row)
        ax1 = plt.subplot(2, 4, col + 1)
        ax1.set_title(f"{cat_name}\nIdx: {idx}\nPred: {pred}, Label: {label}", fontsize=12)
        ax1.imshow(image.astype("uint8"))
        ax1.axis('off')

        # GradCAM overlay (bottom row)
        ax2 = plt.subplot(2, 4, col + 5)
        ax2.set_title("GradCAM", fontsize=12)
        overlay = overlay_heatmap(image.astype("uint8"), heatmap[0])
        ax2.imshow(overlay)
        ax2.axis('off')

    plt.tight_layout()
    plt.show()


# In[51]:


idx_dict = {
    "True Positive": 120,
    "True Negative": 10,
    "False Positive": 0,
    "False Negative": 0
}
plot_gradcam_for_categories(
    images=images,
    binary_preds=binary_preds,
    test_labels=test_labels,
    model=new_model,
    idx_dict=idx_dict
)


# In[58]:


def plot_samples_category(
    images: np.ndarray,
    binary_preds: np.ndarray,
    test_labels: np.ndarray,
    n_show: int = 8
) -> None:
    """
    Plots a grid of sample images from each prediction category:
    True Positives, True Negatives, False Positives, False Negatives.

    Args:
        images (np.ndarray): Array of images, shape (N, H, W, C).
        binary_preds (np.ndarray): Binary predictions, shape (N,).
        test_labels (np.ndarray): Ground truth labels, shape (N,).
        n_show (int): Number of samples to show per category.
    """
    # Identify indices for each category
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
        # Add row label using fig.text
        fig.text(
            0.08,                                        # x-position (left margin)
            1 - (3.5*row + 0.5) / n_rows,                    # y-position (centered per row)
            cat_name,
            va='center', ha='center',
            fontsize=14, fontweight='bold', rotation=90
        )

    plt.suptitle("Sample Images by Prediction Category", fontsize=18, weight='bold')
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])  # Leave room for row labels and title
    plt.show()


# In[57]:


plot_samples_category(
    images=images,
    binary_preds=binary_preds,
    test_labels=test_labels,
    n_show=8
)


# In[ ]:




