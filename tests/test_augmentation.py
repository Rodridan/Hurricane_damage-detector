import pytest
import tensorflow as tf
from riskscope.augmentation import augment_pipeline, apply_augmentation_to_dataset, apply_resize_to_dataset

def create_dummy_dataset(num=16, img_shape=(128, 128, 3), n_classes=2):
    # Generate random uint8 images and integer labels
    images = tf.random.uniform((num, *img_shape), minval=0, maxval=256, dtype=tf.int32)
    labels = tf.random.uniform((num,), minval=0, maxval=n_classes, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(images, tf.uint8), labels)).batch(8)
    return dataset

def test_augment_pipeline_shape_and_type():
    img = tf.random.uniform((128, 128, 3), minval=0, maxval=255, dtype=tf.int32)
    label = tf.constant(1, dtype=tf.int32)
    aug_img, aug_label = augment_pipeline(tf.cast(img, tf.uint8), label)
    # Should be resized to 224x224x3 and remain uint8
    assert aug_img.shape == (224, 224, 3)
    assert aug_img.dtype == tf.uint8
    assert aug_label == label

def test_apply_augmentation_to_dataset_runs():
    ds = create_dummy_dataset()
    aug_ds = apply_augmentation_to_dataset(ds)
    sample_imgs, sample_labels = next(iter(aug_ds))
    assert sample_imgs.shape[1:] == (224, 224, 3)
    assert sample_imgs.dtype == tf.uint8

def test_apply_resize_to_dataset_runs():
    ds = create_dummy_dataset()
    resized_ds = apply_resize_to_dataset(ds)
    sample_imgs, sample_labels = next(iter(resized_ds))
    assert sample_imgs.shape[1:] == (224, 224, 3)
    # Should be float32 after resizing
    assert sample_imgs.dtype == tf.float32

def test_aug_determinism_differs_from_resize():
    # Check that augmentation introduces some variability compared to plain resize
    ds = create_dummy_dataset(num=1)
    aug_img, _ = next(iter(apply_augmentation_to_dataset(ds)))
    resized_img, _ = next(iter(apply_resize_to_dataset(ds)))
    # They are unlikely to be equal
    assert not tf.reduce_all(tf.equal(tf.cast(aug_img, tf.float32), resized_img)), \
        "Augmentation and resize output should not be identical"