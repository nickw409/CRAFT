import keras
from keras.api import layers
import keras.api
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import sys
import tensorflow as tf

try:
  sys.path.append(str(Path('.').resolve().parent))
  import datasets.tusayan_whiteware as tusayan_whiteware
except Exception as e:
  print(e)


class TusNetModel:
    def __init__(
            self, 
            image_dim, 
            batch_size, 
            num_classes, 
            epochs, 
            l2_constant=0.0, 
            use_step_decay=False
    ):
        
        self.image_dim = image_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.model_path = Path('.').resolve() / 'trained_models' / ("tusNet" + ".keras")
        self.history = None
        self.l2_constant = l2_constant
        self.use_step_decay = use_step_decay
        self.model = self.build_model()
   
    def build_model(self):
        inputs = keras.Input(shape=((self.image_dim, self.image_dim) + (3,)))
        x = layers.Rescaling(1./255)(inputs)
        x = self.residual_block(x, filters=32, pooling=True)
        x = self.residual_block(x, filters=64, pooling=True)
        x = self.residual_block(x, filters=128, pooling=True)
        x = self.residual_block(x, filters=256, pooling=True)
        x = self.residual_block(x, filters=512, pooling=False)

        x = layers.GlobalAveragePooling2D()(x)
        #x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def load_model(self):
        self.model = keras.models.load_model(self.model_path)

    def residual_block(self, x, filters, pooling=False):
        residual = x
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
        
        if pooling:
            x = layers.MaxPool2D(pool_size=2, padding="same")(x)
            residual = layers.Conv2D(filters, 1, strides=2)(residual)
        elif filters != residual.shape[-1]:
            residual = layers.Conv2D(filters, 1)(residual)
        x = layers.Add()([x, residual])
        return x
    
    def step_decay(self, epoch):
        init_lr = 0.02
        drop = 0.5
        epochs_drop = 10
        lr = init_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lr
    
    def train(self, train_dataset, val_dataset):
        self.model.compile(optimizer="rmsprop",
                           loss="categorical_crossentropy",
                           metrics=["accuracy"])
        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=self.model_path,
                                            monitor="val_accuracy",
                                            save_best_only=True)
        ]
        if self.use_step_decay:
            callbacks.append(keras.callbacks.LearningRateScheduler(self.step_decay))
        self.history = self.model.fit(train_dataset.prefetch(tf.data.AUTOTUNE),
                                      batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      validation_data=val_dataset.prefetch(tf.data.AUTOTUNE),
                                      callbacks=callbacks)
    
    def evaluate(self, test_dataset):
        print("Evaluating TusNet Model...")
        y_true = []
        y_pred = []
        for image_batch, label_batch in test_dataset:
            y_true.append(label_batch)
            preds = self.model.predict(image_batch)
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

        # plot the accuracy
        plt.style.use("ggplot")
        plt.figure(figsize=(11, 8))
        plt.plot(np.arange(0, self.epochs), self.history.history["accuracy"], label="Train accuracy")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_accuracy"], label="Test accuracy")
        plt.title("model" + " Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        #plt.savefig(full_model_path + "_accuracy_plot.png")

        # plot the training loss
        plt.style.use("ggplot")
        plt.figure(figsize=(11, 8))
        plt.plot(np.arange(0, self.epochs), self.history.history["loss"], label="Train loss")
        plt.plot(np.arange(0, self.epochs), self.history.history["val_loss"], label="Test loss")
        plt.title("model" + " Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        #plt.ylim(0, 2.0)
        plt.legend()
        #plt.savefig(full_model_path  + "_loss_plot.png")
        plt.show()