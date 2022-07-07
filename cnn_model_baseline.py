import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt
from aerial_dataset import dataset_creation

def create_augmentation():
  all_augmentations = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2)
  ])

  return all_augmentations

def model_creation(augmentation):
  model = tf.keras.Sequential()
  model.add(augmentation)
  model.add(tf.keras.layers.Rescaling(1. / 255))

  model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 1)))
  model.add(tf.keras.layers.AveragePooling2D())

  model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
  model.add(tf.keras.layers.AveragePooling2D())

  model.add(tf.keras.layers.Conv2D(filters=26, kernel_size=(3, 3), activation='relu'))
  model.add(tf.keras.layers.AveragePooling2D())

  model.add(tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation='relu'))
  model.add(tf.keras.layers.AveragePooling2D())

  model.add(tf.keras.layers.Conv2D(filters=46, kernel_size=(3, 3), activation='relu'))
  model.add(tf.keras.layers.AveragePooling2D())

  model.add(tf.keras.layers.Flatten())

  model.add(tf.keras.layers.Dense(units=120, activation='relu'))

  model.add(tf.keras.layers.Dense(units=84, activation='relu'))

  model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

  model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

  return model

def model_training(train_ds, val_ds, model):
  batch_size = 32

  earlyStopping = EarlyStopping(monitor='val_accuracy', patience=30, verbose=0, mode='max')
  mcp_save = ModelCheckpoint('.mdl_wts_baseline.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')
  # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

  history = model.fit(train_ds, validation_data=val_ds, batch_size=batch_size, epochs=150,
                      callbacks=[earlyStopping, mcp_save])

  return history

def plot_training(history):

  fig1 = plt.gcf()
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.axis(ymin=0.0, ymax=1)
  plt.grid()
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epochs')
  plt.legend(['train', 'validation'])
  plt.show()

  fig2 = plt.gcf()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.axis(ymin=0.0, ymax=2)
  plt.grid()
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(['train', 'validation'])
  plt.show()


def model_evaluation(val_ds, model):
  # evaluate on test set
  # Loads the weights
  model.load_weights('.mdl_wts_baseline.hdf5')

  # Re-evaluate the model
  loss, acc = model.evaluate(val_ds, verbose=2)
  print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def save_model(model):
  # serialize model to JSON
  model_json = model.to_json()
  with open("model_test.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model_test.h5")
  print("Saved model to disk")


def main():
  train_ds, val_ds = dataset_creation(224, 224, 8)
  augmentations = create_augmentation()
  model = model_creation(augmentations)
  train_history = model_training(train_ds, val_ds, model)
  plot_training(train_history)
  model_evaluation(val_ds, model)
  save_model(model)


if __name__ == '__main__':
  main()

