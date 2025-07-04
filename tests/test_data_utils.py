import os
import pytest
import tensorflow as tf
import gdown

from riskscope.data_utils import (
    prepare_train_and_val_datasets,
    get_test_dataset,
    eval_model_on_test
)
from riskscope.config import IMG_DIMS, BATCH_SIZE, EVAL_BATCH_SIZE, CLASSES, TRAIN_SUBDIR,TEST_SUBDIR, DATA_DIR

TRAIN_DIR = os.path.join(DATA_DIR, TRAIN_SUBDIR)
TEST_DIR = os.path.join(DATA_DIR, TEST_SUBDIR)

@pytest.mark.skipif(not os.path.exists(TRAIN_DIR), reason="Train directory not found.")
def test_prepare_train_and_val_datasets_runs():
    train_ds, val_ds = prepare_train_and_val_datasets(
        train_dir=TRAIN_DIR,
        img_dims=IMG_DIMS,
        batch_size=BATCH_SIZE,
        class_names=CLASSES
    )
    # Should return two tf.data.Dataset objects
    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(val_ds, tf.data.Dataset)
    # Check if not empty
    train_batch = next(iter(train_ds))
    val_batch = next(iter(val_ds))
    assert train_batch[0].shape[1:3] == IMG_DIMS

@pytest.mark.skipif(not os.path.exists(TEST_DIR), reason="Test directory not found.")
def test_get_test_dataset_runs():
    test_ds = get_test_dataset(
        test_dir=TEST_DIR,
        img_dims=IMG_DIMS,
        batch_size=EVAL_BATCH_SIZE,
        class_names=CLASSES
    )
    # Should return a tf.data.Dataset
    assert isinstance(test_ds, tf.data.Dataset)
    test_batch = next(iter(test_ds))
    assert test_batch[0].shape[1:3] == (224, 224)  # model input size

@pytest.mark.skipif(not os.path.exists(TEST_DIR), reason="Test directory not found.")
def test_eval_model_on_test_dummy():
    # Use a dummy model for test (does nothing, always outputs zeros)
    class DummyModel:
        def predict(self, x):
            return tf.zeros((x.shape[0], 1))
    model = DummyModel()
    test_ds, test_labels, predictions = eval_model_on_test(
        model=model,
        test_dir=TEST_DIR,
        img_dims=IMG_DIMS,
        batch_size=EVAL_BATCH_SIZE,
        class_names=CLASSES
    )
    assert len(test_labels) == len(predictions)
