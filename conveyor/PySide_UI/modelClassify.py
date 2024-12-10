import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#define constants
IMAGE_DIM = 384
MODEL_PATH = Path('.') / 'convNext.tflite'
CLASS_NAMES = ['Kanaa', 'Black_Mesa', 'Sosi', 'Dogoszhi', 'Flagstaff', 'Tusayan', 'Kayenta']

#get images from specified directory
def load_dataset(directory, target_size):
    image_paths = [str(p) for p in Path(directory).glob('**/*') if p.is_file()]
    images = []
    for image_path in image_paths:
        #load and preprocess image
        img = load_img(image_path, target_size=(target_size, target_size))
        #normalize pixel values
        img_array = img_to_array(img) / 255.0  
        #save image with its path
        images.append((img_array, image_path))  
    return images

#start up tflite
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

#use our tflite model to get results
def predict_with_tflite_model(interpreter, input_details, output_details, images):
    results = []
    for img_array, img_path in images:
        #add batch dimension
        input_data = np.expand_dims(img_array, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        #get the prediction
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class = np.argmax(predictions)
        results.append((img_path, predicted_class, predictions))
    return results

def display_results(results, class_names, num_to_display=5):
    for i, (img_path, predicted_class, predictions) in enumerate(results[:num_to_display]):
        print(f"File: {img_path}")
        print(f"Predicted Class: {class_names[predicted_class]}")
        print(f"Prediction Probabilities: {predictions}")
        print("-" * 50)  # separator for better readability


# callable function to classify sherds
def classifySherds(dataPath):

    DATASET_PATH = Path('.') / dataPath

    # load the tflite model
    interpreter, input_details, output_details = load_tflite_model(MODEL_PATH)
    print("TFLite model loaded successfully.")
    
    # load dataset
    dataset = load_dataset(DATASET_PATH, IMAGE_DIM)
    print(f"Loaded {len(dataset)} images from {DATASET_PATH}")
    
    # process images with the tflite model
    predictions = predict_with_tflite_model(interpreter, input_details, output_details, dataset)
    print("Inference complete.")
    
    # display a few results
    display_results(predictions, CLASS_NAMES, num_to_display=5)
