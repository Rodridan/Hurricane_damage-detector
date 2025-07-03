# augmentation.py
import tensorflow as tf
from typing import Tuple
from loguru import logger

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