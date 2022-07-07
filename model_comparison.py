import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from cf_matrix import make_confusion_matrix


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


def revaluate_model(model_json_file, model_h5_file, val_ds):
    # load json and create model
    json_file = open(model_json_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_h5_file)
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    loss, score = loaded_model.evaluate(val_ds, verbose=2)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score * 100))
    print(loaded_model.metrics_names[0], "{:10.4f}".format(loss))
    # print("{:10.4f}".format(x))
    return loaded_model


def plot_confusion_matrix(val_ds, loaded_model, file_name):
    y_pred = []  # store predicted labels
    y_true = []  # store true labels
    # iterate over the dataset
    for image_batch, label_batch in val_ds:  # use dataset.unbatch() with repeat
        # append true labels
        y_true.append(label_batch)
        # compute predictions
        preds = loaded_model.predict(image_batch)
        # append predicted labels
        y_pred.append(np.argmax(preds, axis=- 1))

    # convert the true and predicted labels into tensors
    correct_labels = tf.concat([item for item in y_true], axis=0)
    predicted_labels = tf.concat([item for item in y_pred], axis=0)
    # one hot encode
    encoded = to_categorical(predicted_labels, num_classes=3)
    # convert to tensors
    predictions = tf.convert_to_tensor(encoded)
    # convert to value array
    y_true = tf.math.argmax(correct_labels, 1)
    y_pred = tf.math.argmax(predictions, 1)
    # create confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    categories = ['Ocean', 'Ships', 'Unknown Floating Objects']
    make_confusion_matrix(cf_matrix,
                          group_names=labels,
                          categories=categories,
                          figsize=(10, 6),
                          cmap='binary')
    # save to file and show
    plt.savefig(file_name + ".png", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    batch_size = 8
    img_height = 224
    img_width = 224
    train_ds, val_ds = dataset_creation()
    baseline_model = revaluate_model("model_baseline.json", "model_baseline.h5", val_ds)
    tuned_model_one = revaluate_model("model_tuned_one.json", "model_tuned_one.h5", val_ds)
    tuned_model_two = revaluate_model("model_tuned_two.json", "model_tuned_two.h5", val_ds)
    plot_confusion_matrix(val_ds, baseline_model, "confusion_baseline.png")
    plot_confusion_matrix(val_ds, tuned_model_one, "confusion_tuned_one.png")
    plot_confusion_matrix(val_ds, tuned_model_two, "confusion_tuned_two.png")

