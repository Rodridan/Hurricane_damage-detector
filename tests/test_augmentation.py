from damage_detector.augmentation import augment_pipeline
import tensorflow as tf

def test_augmentation():
    # Create dummy data
    img = tf.zeros((128,128,3), dtype=tf.uint8)
    label = tf.constant(1)
    img_aug, label_aug = augment_pipeline(img, label)
    print("Augmented image shape:", img_aug.shape)

if __name__ == "__main__":
    test_augmentation()
