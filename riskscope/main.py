import os

from riskscope.config import ( GDRIVE_ID, DATA_DIR, TRAIN_SUBDIR, TEST_SUBDIR, CLASSES, BATCH_SIZE, USE_PRETRAINED, PRETRAINED_MODEL_PATH)
from riskscope.data_utils import (download_and_extract_data, prepare_train_and_val_datasets, eval_model_on_test)
from riskscope.augmentation import (apply_augmentation_to_dataset, apply_resize_to_dataset)
from riskscope.model import build_resnet_model, train_model
from riskscope.model_loader import load_pretrained_model
from riskscope.eval_viz import (plot_classification_summary, extract_images_and_labels,
    plot_false_negatives_positives, plot_gradcam_for_categories, plot_samples_category)
from riskscope.logging_setup import setup_logging

from loguru import logger

# ========== MAIN PIPELINE ==========

if __name__ == "__main__":
    setup_logging()
    logger.info("=== Hurricane Damage Detector Pipeline Started ===")

    # Download and extract data if not present
    extract_path = download_and_extract_data(
        gdrive_id=GDRIVE_ID,
        data_dir=DATA_DIR,
        zip_filename="hurricane_detector.zip",
        extracted_folder_name=TRAIN_SUBDIR
    )
    TRAIN_DIR = os.path.join(DATA_DIR, TRAIN_SUBDIR)
    TEST_DIR = os.path.join(DATA_DIR, TEST_SUBDIR)

    # --- Prepare datasets ---
    train_ds, val_ds = prepare_train_and_val_datasets(train_dir=TRAIN_DIR, batch_size=BATCH_SIZE)
    train_ds = apply_augmentation_to_dataset(train_ds)
    val_ds = apply_resize_to_dataset(val_ds)

    # ========== MODEL LOGIC ==========
    if USE_PRETRAINED and os.path.exists(PRETRAINED_MODEL_PATH):
        logger.info(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}")
        model = load_pretrained_model(PRETRAINED_MODEL_PATH)
        history1 = None  # No training history
        logger.info("Model loaded; skipping training phase.")
    else:
        # --- PHASE 1: Train only custom layers (base frozen) ---
        model = build_resnet_model(input_shape=(224, 224, 3), learning_rate=1e-4, train_base=False)
        logger.info("Phase 1: Training custom layers (base frozen)")
        history1 = train_model(model, train_ds, val_ds, epochs=10)

        # Save phase 1 model (optional)
        # model.save("pretrained_models/hurricane_resnet50_224x224_b32_e10_ph1_stdaug_YYYYMMDD.keras")

    # --- PHASE 1: Evaluation & Visualization ---
    logger.info("Evaluating model after phase 1")
    test_ds, test_labels, predictions = eval_model_on_test(model, test_dir=TEST_DIR)
    binary_preds = (predictions > 0.5).astype(int)
    plot_classification_summary(
        predictions=predictions, test_labels=test_labels,
        histories=[h for h in [history1] if h is not None],
        class_names=CLASSES, title="Classification Results: Phase 1",
        filename_base="classification_summary_phase1"
    )
    images, labels = extract_images_and_labels(test_ds)
    plot_false_negatives_positives(
        images=images, test_labels=test_labels, binary_preds=binary_preds, n_show=8,
        title="Misclassifications: Phase 1", filename_base="fp_fn_grid_phase1"
    )
    plot_samples_category(
        images=images, binary_preds=binary_preds, test_labels=test_labels, n_show=8,
        filename_base="category_samples_phase1"
    )

    # ========== PHASE 2: Fine-tuning (optional, skip if USE_PRETRAINED and you only want inference) ==========
    if not USE_PRETRAINED:
        logger.info("Phase 2: Fine-tuning last block (except BN layers)")
        set_trainable_layers(model, trainable_base_layers=["conv5"], freeze_bn=True)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        history2 = train_model(model, train_ds, val_ds, epochs=10)

        # --- PHASE 2: Evaluation & Visualization ---
        logger.info("Evaluating model after phase 2 (fine-tuning)")
        test_ds, test_labels, predictions = eval_model_on_test(model, test_dir=TEST_DIR)
        binary_preds = (predictions > 0.5).astype(int)
        plot_classification_summary(
            predictions=predictions, test_labels=test_labels,
            histories=[h for h in [history1, history2] if h is not None],
            class_names=CLASSES, title="Classification Results: Both Phases",
            filename_base="classification_summary_phases"
        )
        images, labels = extract_images_and_labels(test_ds)
        plot_false_negatives_positives(
            images=images, test_labels=test_labels, binary_preds=binary_preds, n_show=8,
            title="Misclassifications: Phase 2", filename_base="fp_fn_grid_phase2"
        )
        plot_samples_category(
            images=images, binary_preds=binary_preds, test_labels=test_labels, n_show=8,
            filename_base="category_samples_phase2"
        )
        # (Optional) Save fine-tuned model
        # model.save("pretrained_models/hurricane_resnet50_224x224_b32_e10_ph2_ft_stdaug_YYYYMMDD.keras")

    # --- GradCAM (optional, after last phase) ---
    idx_dict = {"True Positive": 5, "True Negative": 10, "False Positive": 0, "False Negative": 0}
    plot_gradcam_for_categories(
        images=images, binary_preds=binary_preds, test_labels=test_labels,
        model=model, idx_dict=idx_dict, filename_base="gradcam_categories"
    )

    logger.success("=== Pipeline completed successfully! ===")
