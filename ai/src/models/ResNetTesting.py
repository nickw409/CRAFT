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

import craft.datasets.tusayan_whiteware as tusayan_whiteware


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



def train_model(model, train_dataset, steps_per_epoch=False):   
    # Prevent base model from training
    for layer in baseModel.layers:
        layer.trainable = False

    opt = keras.optimizers.RMSprop(learning_rate=0.01)

    model.compile(loss="categorical_crossentropy", optimizer=opt,
                metrics=["accuracy"])

    # Add Learning rate scheduler to model callbacks
    callbacks=[keras.callbacks.LearningRateScheduler(step_decay)]

    print("[INFO] training head...")

    if steps_per_epoch:
        model.fit(train_dataset, batch_size=batch_size, epochs=epochs_head, 
            validation_split=0.1, validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // batch_size, callbacks=callbacks)
    else:    
        model.fit(train_dataset, batch_size=batch_size, epochs=epochs_head, 
            validation_split=0.1, validation_data=(x_test, y_test),
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

    

    full_model_path = os.path.join(models_dir, "", "ResNet152V2" + ".keras")
    callbacks=[keras.callbacks.LearningRateScheduler(step_decay),
               keras.callbacks.ModelCheckpoint(full_model_path, monitor="val_accuracy",
                                               verbose=1, save_best_only=True,
                                               save_weights_only=False, mode="max",
                                               save_freq="epoch"),
                keras.callbacks.EarlyStopping(patience=10)]

    print("Fine-Tuning Final Model...")
    
    if steps_per_epoch:
        model.fit(train_dataset, batch_size=batch_size, epochs=epochs, 
            validation_split=0.1, validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // batch_size, callbacks=callbacks)
    else:    
        model.fit(train_dataset, batch_size=batch_size, epochs=epochs, 
            validation_split=0.1, validation_data=(x_test, y_test),
            callbacks=callbacks)

    model_best = keras.models.load_model(full_model_path)
    return model_best


# Arg parser for set number
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--set", required=True, 
  help="integer index of sherd train/test set")
args = vars(ap.parse_args())

set_number = str(args["set"])
images_dir = os.getcwd() + "/image_data"
num_classes = 7
image_dimension = 224
batch_size = 32
epochs_head = 10
epochs = 50
l2_constant = 0.02
input = keras.Input(shape=(image_dimension, image_dimension, 3))

models_dir = os.getcwd() + "/image_data" + "/Set_" + set_number + "/models"

(train_data,train_labels) = tusayan_whiteware.load_data(set_number, image_dimension, verbose=250)

(test_data,test_labels) = tusayan_whiteware.load_data(set_number, image_dimension,verbose=250)

# Set classNames from train_labels list, rename to eliminate numbers from the front
classNames = [str(x) for x in np.unique(train_labels)]
classNames = ['Kanaa',  'Black_Mesa', 'Sosi', 'Dogoszhi', 'Flagstaff', 'Tusayan', 'Kayenta']


train_data = train_data.astype("float")
train_data = resnet_v2.preprocess_input(train_data)

print("Training data loaded")

# Load test data separately
test_data = test_data.astype("float") 
test_data = resnet_v2.preprocess_input(test_data)
print("Test data loaded")

(x_train, x_test, y_train, y_test) = (train_data,test_data,train_labels,test_labels)

# convert the labels from integers to vectors
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)


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


data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(factor=0.5, fill_mode="constant", 
                                    fill_value=1.0),
    keras.layers.RandomZoom(height_factor=0.3, width_factor=0.3,
                            fill_mode="constant", fill_value=1.0),
])

# Testing if ImageDataGenerator against augmentation layers
#Parameters for ImageDataGenerator
shift=0.0
zoom=0.3
# construct the image generator for data augmentation
#fill_mode is value put into empty spaces created by rotation or zooming; cval=1.0 means white
aug = ImageDataGenerator(rotation_range=180,
	horizontal_flip=False, vertical_flip=False, width_shift_range=shift, 
    height_shift_range=shift, zoom_range=zoom, fill_mode="constant",cval=1.0)

generator_augmented_train_dataset = tf.data.Dataset.from_generator(
    lambda: aug.flow(x_train, y_train, batch_size=batch_size),
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None), 
        tf.TensorSpec(shape=(None, 7), dtype=tf.int64, name=None)
    )
)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

layer_augmented_train_dataset = (
    train_dataset
    .shuffle(batch_size*100)
    .batch(batch_size)
    .map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

train_dataset = (
    train_dataset
    .shuffle(batch_size*100)
    .batch(batch_size)
)

best_generator_augmented_model = train_model(model, generator_augmented_train_dataset,
                                             steps_per_epoch=True)
best_layer_augmented_model = train_model(model, layer_augmented_train_dataset)
best_base_model = train_model(model, train_dataset)

# evaluate test data
print("[INFO] evaluating test data for generator-augmented model...")
predictions = best_generator_augmented_model.predict(x_test, batch_size=batch_size)
class_report=classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)
print(class_report)
print("Confusion matrix")
print(classNames)
con_mat=confusion_matrix(np.argmax(y_test, axis=1), predictions.argmax(axis=1))
print(con_mat)


print("[INFO] evaluating test data for layer-augmented model...")
predictions = best_layer_augmented_model.predict(x_test, batch_size=batch_size)
class_report=classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)
print(class_report)
print("Confusion matrix")
print(classNames)
con_mat=confusion_matrix(np.argmax(y_test, axis=1), predictions.argmax(axis=1))
print(con_mat)

print("[INFO] evaluating test data for base model...")
predictions = best_base_model.predict(x_test, batch_size=batch_size)
class_report=classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)
print(class_report)
print("Confusion matrix")
print(classNames)
con_mat=confusion_matrix(np.argmax(y_test, axis=1), predictions.argmax(axis=1))
print(con_mat)