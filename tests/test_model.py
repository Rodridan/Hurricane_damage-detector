import pytest
import tensorflow as tf
from riskscope.model import build_resnet_model, train_model

def test_build_resnet_model_default():
    model = build_resnet_model()
    # Check it's a tf.keras.Model
    assert isinstance(model, tf.keras.Model)
    # Input and output shapes
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)
    # Model should be compiled
    assert model.optimizer is not None
    assert model.loss == 'binary_crossentropy'

def test_build_resnet_model_trainable():
    model = build_resnet_model(train_base=True)
    trainable_layers = [layer.trainable for layer in model.layers]
    assert any(trainable_layers), "All layers should not be frozen if train_base=True"

def test_build_resnet_model_frozen():
    model = build_resnet_model(train_base=False)
    # All layers except the custom head should be frozen
    base_trainables = [l.trainable for l in model.layers if 'resnet' in l.name or 'conv' in l.name]
    assert not any(base_trainables), "Base layers should be frozen if train_base=False"

def test_train_model_runs(monkeypatch):
    # Dummy dataset
    input_shape = (32, 224, 224, 3)
    x = tf.random.uniform(input_shape)
    y = tf.random.uniform((32, 1), minval=0, maxval=2, dtype=tf.int32)
    train_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)
    val_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)

    model = build_resnet_model()

    # Patch model.fit to avoid heavy computation
    def dummy_fit(*args, **kwargs):
        class DummyHistory:
            history = {'accuracy': [1.0], 'val_accuracy': [1.0]}
        return DummyHistory()
    monkeypatch.setattr(model, "fit", dummy_fit)
    history = train_model(model, train_ds, val_ds, epochs=1)
    assert hasattr(history, 'history')

