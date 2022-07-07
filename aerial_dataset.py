import tensorflow as tf

def dataset_creation(img_height, img_width, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "C:/aerial_images",
        validation_split=0.2,
        subset="training",
        label_mode="categorical",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "C:/aerial_images",
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds