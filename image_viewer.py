import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def dataset_creation():
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


def augmentation_viewer(dataset, augmentations):
  class_names = dataset.class_names
  for images, labels in dataset.take(1):
    for i in range(batch_size):
      # plot original
      plt.figure(figsize=(10, 10))
      # ax = plt.subplot(6, 6, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[np.argmax(labels[i])])
      plt.axis("off")
      plt.show()
      # plot augmented
      plt.figure(figsize=(10, 10))
      augmented_image = augmentations(images[i])
      plt.imshow(augmented_image.numpy().astype("uint8"))
      plt.title(class_names[np.argmax(labels[i])])
      plt.axis("off")
      plt.show()


def create_augmentation():
  all_augmentations = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2)
  ])

  return all_augmentations


if __name__ == '__main__':
  batch_size = 1
  img_height = 224
  img_width = 224
  train_ds, val_ds = dataset_creation()
  augmentations = create_augmentation()
  augmentation_viewer(train_ds, augmentations)
