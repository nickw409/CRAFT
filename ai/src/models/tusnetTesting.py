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

from tusNetModel import TusNetModel
try:
  sys.path.append(str(Path('.').resolve().parent))
  import datasets.tusayan_whiteware as tusayan_whiteware
except Exception as e:
  print(e)


image_dim = 224
num_classes = 7
batch_size = 32
epochs = 100
l2_constant = 0.02

data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(factor=0.5, 
                                fill_mode="constant", 
                                fill_value=1.0),
    keras.layers.RandomZoom(height_factor=(-0.2, 0.2), 
                            width_factor=(-0.2, 0.2),
                            fill_mode="constant", 
                            fill_value=1.0),
])

(train_dataset, test_dataset) = tusayan_whiteware.load_data(
   image_dimension=(image_dim, image_dim),
   training_split=(0.8,0.0,0.2),
   batch_size=batch_size
   )

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y),
                                  num_parallel_calls=tf.data.AUTOTUNE)

model = TusNetModel(image_dim=image_dim,
                    batch_size=batch_size,
                    num_classes=num_classes,
                    epochs=epochs,
                    l2_constant=l2_constant)
model.train(train_dataset=train_dataset,
            val_dataset=test_dataset)

model.load_model()
model.evaluate(test_dataset=test_dataset)