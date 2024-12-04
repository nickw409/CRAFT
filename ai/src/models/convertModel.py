import tensorflow as tf
import keras
import os;os.environ["TF_USE_LEGACY_KERAS"]="1"
import shutil

keras_model = tf.keras.models.load_model('trained_models/convNext.keras')
#keras_model.export('trained_models/convNext')
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

with open('trained_models/downgraded_convNext.tflite', 'wb') as f:
    f.write(tflite_model)

shutil.rmtree('trained_models/convNext')