import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
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

# load json and create model
json_file = open('model_baseline.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_baseline.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
loss, score = loaded_model.evaluate(val_ds, verbose=2)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score * 100))
print(loaded_model.metrics_names[0], "{:10.4f}".format(loss))
#print("{:10.4f}".format(x))

# load json and create model
json_file = open('model_tuned_one.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_tuned_one.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
loss, score = loaded_model.evaluate(val_ds, verbose=2)
print("Tuned - One additional layer")
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score * 100))
print(loaded_model.metrics_names[0], "{:10.4f}".format(loss))
