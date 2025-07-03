# model.py
from tensorflow.keras import Model, layers
from loguru import logger
from typing import Tuple, Sequence
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization


#--------------------------------------------------------------
# MODEL DEFINITION & TRAINING
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
