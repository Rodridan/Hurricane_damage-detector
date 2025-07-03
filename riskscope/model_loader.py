import tensorflow as tf
from loguru import logger
from typing import Tuple, Sequence

def load_pretrained_model(model_path, custom_objects=None):
    """
    Load a Keras model from .keras or .h5 file.

    Args:
        model_path (str): Path to the saved model file.
        custom_objects (dict, optional): Any custom layers/objects if used.

    Returns:
        tf.keras.Model: Loaded model.
    """
    logger.info(f"Loading pretrained model from {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    logger.success(f"Loaded pretrained model from {model_path}")
    return model
