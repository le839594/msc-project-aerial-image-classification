import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import matplotlib.pyplot as plt

batch_size = 8
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
print(class_names)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])

def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(data_augmentation)
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
    hp_units_1 = hp.Int('units1', min_value=88, max_value=248, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units_1, activation='relu'))
    hp_units_2 = hp.Int('units2', min_value=32, max_value=128, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units_2, activation='relu'))

    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))



    model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

    return model

batch_size = 32
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=100,
                     factor=12,
                     hyperband_iterations=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
#mcp_save = ModelCheckpoint('.mdl_wts_v2.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

tuner.search(train_ds, validation_data = val_ds, batch_size=batch_size, epochs=150, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units1')} and the optimal number in the second layer 
is {best_hps.get('units2')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_ds, validation_data = val_ds, batch_size=batch_size, epochs=150)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

mcp_save = ModelCheckpoint('.mdl_wts_v3.hdf5', save_best_only=True, monitor='val_accuracy', mode='max')

# Retrain the model
history = hypermodel.fit(train_ds, validation_data = val_ds, batch_size=batch_size, epochs=best_epoch, callbacks=[mcp_save])

fig1 = plt.gcf()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.axis(ymin=0.0,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

fig2 = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.axis(ymin=0.0,ymax=2)
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()

# evaluate on test set
# Loads the weights
hypermodel.load_weights('.mdl_wts_v3.hdf5')

# Re-evaluate the model
loss, acc = hypermodel.evaluate(val_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))