import keras
from keras.api.applications import ResNet152V2, resnet_v2
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from csv import reader
import matplotlib.pyplot as plt
import os
from random import shuffle
from pathlib import Path
import sys

try:
  sys.path.append(str(Path('.').resolve().parent))
  import datasets.tusayan_whiteware as tusayan_whiteware
except Exception as e:
  print(e)


def preprocess(images, labels):
    return resnet_v2.preprocess_input(images), labels


def step_decay(epoch):
    initAlpha=.0020
    if epoch <= 20:
        alpha=initAlpha
    elif epoch <= 40:
        alpha=initAlpha*0.75
    elif epoch <= 60:
        alpha=initAlpha*0.50
    elif epoch <= 80:
        alpha=initAlpha*0.25
    elif epoch <=100:
        alpha=initAlpha*0.175
    elif epoch <= 125:
        alpha=initAlpha*0.10
    else:
        alpha=initAlpha*0.05
    return float(alpha)


def train_model(model, train_dataset, val_dataset):   
    # Prevent base model from training
    for layer in baseModel.layers:
        layer.trainable = False

    opt = keras.optimizers.RMSprop(learning_rate=0.01)

    model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

    # Add Learning rate scheduler to model callbacks
    callbacks=[keras.callbacks.LearningRateScheduler(step_decay)]

    print("[INFO] training head...")
    model.fit(train_dataset, batch_size=batch_size, epochs=epochs_head, 
        validation_split=0.1, validation_data=val_dataset,
        callbacks=callbacks)

    # Allow base model to train along with head model for full training
    for layer in baseModel.layers[0:]:
        layer.trainable = True

    print("Re-compiling model...")

    # Have to rebuild optimizer for model recompile
    opt = keras.optimizers.SGD(learning_rate=0.005)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])
    print("Model Compiled")

    full_model_path = Path('.').resolve() / 'trained_models' / ("ResNet152V2" + ".keras")
    log_path = Path('.').resolve() / 'logs'
    if not log_path.exists():
        log_path.mkdir()
    tensorboard = keras.callbacks.TensorBoard(log_dir=str(log_path))
    callbacks=[keras.callbacks.LearningRateScheduler(step_decay),
               keras.callbacks.ModelCheckpoint(full_model_path, monitor="val_accuracy",
                                               verbose=1, save_best_only=True,
                                               save_weights_only=False, mode="max",
                                               save_freq="epoch"),
                keras.callbacks.EarlyStopping(patience=3),
                tensorboard,
                ]

    print("Fine-Tuning Final Model...")
    model.fit(train_dataset, batch_size=batch_size, epochs=epochs, 
        validation_split=0.1, validation_data=val_dataset,
        callbacks=callbacks)

    model_best = keras.models.load_model(full_model_path)
    return model_best


image_dir = Path('.').resolve().parents[1] / 'image_data'
num_classes = 7
image_dimension = 224
batch_size = 32
epochs_head = 10
epochs = 50
l2_constant = 0.02
input = keras.Input(shape=(image_dimension, image_dimension, 3))

(train_ds, test_ds) = tusayan_whiteware.load_data(
   image_dimension=(image_dimension, image_dimension),
   training_split=(0.8,0.0,0.2),
   batch_size=batch_size
   )

train_ds = train_ds.map(preprocess)
#val_ds = val_ds.map(preprocess)
test_ds = test_ds.map(preprocess)

# Create base model of ResNet
baseModel = ResNet152V2(weights="imagenet", include_top=False, 
                        input_tensor=input)

# Create head model from ResNet base model for transfer learning
headModel = baseModel.output
headModel = keras.layers.GlobalAveragePooling2D()(headModel)
headModel = keras.layers.Flatten()(headModel)
headModel = keras.layers.Dense(1024, activation="relu", 
                               kernel_regularizer=keras.regularizers.L2(l2_constant))(headModel)
headModel = keras.layers.Dense(num_classes, activation="softmax")(headModel)

model = keras.models.Model(inputs=baseModel.input, outputs=headModel)

best_base_model = train_model(model, 
                              train_dataset=train_ds, 
                              val_dataset=test_ds)

print("[INFO] evaluating test data for base model...")
y_true = []
y_pred = []
for image_batch, label_batch in test_ds:
    y_true.append(label_batch)
    preds = best_base_model.predict(image_batch)
    y_pred.append(np.argmax(preds, axis=-1))
correct_labels = tf.concat([item for item in y_true], axis = 0)
predicted_labels = tf.concat([item for item in y_pred], axis = 0)
class_report=classification_report(np.argmax(correct_labels, axis=1),
                                   predicted_labels,
                                   target_names=[label.name for label in tusayan_whiteware.SherdType])
print(class_report)
print("Confusion matrix")
print([label.name for label in tusayan_whiteware.SherdType])
con_mat=confusion_matrix(np.argmax(correct_labels, axis=1), predicted_labels)
print(con_mat)
