import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

batch_size = 6
img_height = 224
img_width = 224


train_ds = tf.keras.utils.image_dataset_from_directory(
  "C:/aerial_images",
  validation_split=0.2,
  subset="training",
  label_mode="categorical",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds =tf.keras.utils.image_dataset_from_directory(
  "C:/aerial_images",
  validation_split=0.2,
  subset="validation",
  label_mode="categorical",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)



class_names = train_ds.class_names

all_augmentations = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2)
])

### To visualize the images

for images, labels in train_ds.take(1):
  for i in range(batch_size):
    plt.figure(figsize=(10, 10))
    #ax = plt.subplot(6, 6, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[np.argmax(labels[i])])
    plt.axis("off")
    plt.show()
    plt.figure(figsize=(10, 10))
    augmented_image = all_augmentations(images[i])
    plt.imshow(augmented_image.numpy().astype("uint8"))
    plt.title(class_names[np.argmax(labels[i])])
    plt.axis("off")
    plt.show()




