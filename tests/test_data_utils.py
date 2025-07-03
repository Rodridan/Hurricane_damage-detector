from riskscope.data_utils import download_and_extract_data, prepare_train_and_val_datasets

def test_download():
    # Should not re-download if already present
    path = download_and_extract_data(
        gdrive_id="1pByxsenTnJGBKnKhLTXBqbUN_Kbm7PNK",
        data_dir="data",
        zip_filename="hurricane_detector.zip",
        extracted_folder_name="train_hurricane"
    )
    print("Download and extraction successful:", path)

def test_dataset_loading():
    train_ds, val_ds = prepare_train_and_val_datasets(train_dir="data/train_hurricane")
    for batch in train_ds.take(1):
        print("Loaded one batch from train set. Batch shape:", batch[0].shape)
    for batch in val_ds.take(1):
        print("Loaded one batch from val set. Batch shape:", batch[0].shape)

if __name__ == "__main__":
    test_download()
    test_dataset_loading()