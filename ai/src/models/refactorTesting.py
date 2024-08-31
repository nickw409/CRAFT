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
import argparse
from random import shuffle
from pathlib import Path
import sys

#import src.datasets.tusayan_whiteware as tusayan_whiteware
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


image_dir = Path('.').resolve().parents[1] / 'image_data'
num_classes = 7
image_dimension = 224
batch_size = 32
epochs_head = 10
epochs = 50
l2_constant = 0.02
input = keras.Input(shape=(image_dimension, image_dimension, 3))

(train_ds, val_ds, test_ds) = tusayan_whiteware.load_data(
   image_dimension=(image_dimension, image_dimension),
   training_split=(0.6,0.2,0.2),
   batch_size=batch_size
   )

train_ds = train_ds.map(preprocess)
val_ds = val_ds.map(preprocess)

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

# Prevent base model from training
for layer in baseModel.layers:
   layer.trainable = False

opt = keras.optimizers.RMSprop(learning_rate=0.01)

model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])

# Add Learning rate scheduler to model callbacks
callbacks=[keras.callbacks.LearningRateScheduler(step_decay)]

print("[INFO] training head...")

model.fit(train_ds, batch_size=batch_size, epochs=epochs_head, 
   validation_data=val_ds, callbacks=callbacks)

# Allow base model to train along with head model for full training
for layer in baseModel.layers[0:]:
   layer.trainable = True

print("Re-compiling model...")

# Have to rebuild optimizer for model recompile
opt = keras.optimizers.SGD(learning_rate=0.005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
            metrics=["accuracy"])
print("Model Compiled")

callbacks=[keras.callbacks.LearningRateScheduler(step_decay)]

print("Fine-Tuning Final Model...")
  
model.fit(train_ds, batch_size=batch_size, epochs=epochs, 
   validation_data=val_ds, callbacks=callbacks)