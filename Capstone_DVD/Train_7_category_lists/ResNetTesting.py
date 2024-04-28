import keras
from keras.applications import ResNet152V2, resnet_v2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import cv2
from csv import reader
import os
import argparse
from random import shuffle


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


def create_data_arrays_list(imageData, image_dimension, verbose=-1):
    # initialize the list of features and labels
    data = []
    labels = []
    i=0

    # loop over the input images
    for image_input in imageData:
        # load the image
        images_dir = os.getcwd() + "/image_data/images/"
        image = cv2.imread(images_dir + image_input[0][:])
        

        
        #image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


        # resize image, convert image to grayscale, then back to original color scheme
        image = cv2.cvtColor(cv2.cvtColor(cv2.resize(image, (image_dimension, image_dimension),
            interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        
        #convert image into array format using Keras function
        image=keras.utils.img_to_array(image)

        #add image data to array
        data.append(image)
        
        #Change label names so that class numbers will correspond to chronological sequence; class names changed back further on
        if image_input[:][1] == 'Kanaa': append_label='00'+'Kanaa'
        elif image_input[:][1] == 'Black_Mesa': append_label='01'+'Black_Mesa'
        elif image_input[:][1] == 'Sosi': append_label='02'+'Sosi'
        elif image_input[:][1] == 'Dogoszhi': append_label='03'+'Dogoszhi'
        elif image_input[:][1] == 'Flagstaff': append_label='04'+'Flagstaff'
        elif image_input[:][1] == 'Tusayan': append_label='05'+'Tusayan'
        else: append_label='06'+'Kayenta'
        
        #write label name
        labels.append(append_label)

        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1,
            len(imageData)))
        i=i+1

    # return image array data, labels
    return (np.array(data), np.array(labels))

# Arg parser for set number
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--set", required=True,
	help="integer index of sherd train/test set")
args = vars(ap.parse_args())

set_number=str(args["set"])
set_directory= os.getcwd() + "/image_data/"
images_dir = os.getcwd() + "/image_data/"
num_classes = 7
image_dimension = 224
batch_size = 32
epochs_head = 10
epochs = 90
l2_constant=0.02
input = keras.Input(shape=(image_dimension, image_dimension, 3))

#Define directories to use based on set number for loading data, saving data
train_dataset=set_directory + "/Set_" + set_number +"/train_" + set_number + ".csv"
test_dataset=set_directory + "/Set_" + set_number +"/test_" + set_number + ".csv"

# Get lists of train images, types from file
print("[INFO] loading images...")
with open(train_dataset, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    train_data_list = list(csv_reader)

#close file
read_obj.close()

#randomize order of images
shuffle(train_data_list)

#same for test images, types
with open(test_dataset, 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    test_data_list = list(csv_reader)


read_obj.close()

(train_data,train_labels)=create_data_arrays_list(train_data_list,image_dimension, verbose=250)

(test_data,test_labels)=create_data_arrays_list(test_data_list,image_dimension,verbose=250)

#Set classNames from train_labels list, rename to eliminate numbers from the front
classNames = [str(x) for x in np.unique(train_labels)]
classNames = ['Kanaa',  'Black_Mesa', 'Sosi', 'Dogoszhi', 'Flagstaff', 'Tusayan', 'Kayenta']


train_data = train_data.astype("float")
train_data = resnet_v2.preprocess_input(train_data)

print("Training data loaded")

#Load test data separately
test_data = test_data.astype("float") 
test_data = resnet_v2.preprocess_input(test_data)
print("Test data loaded")

(x_train, x_test, y_train, y_test) = (train_data,test_data,train_labels,test_labels)

# convert the labels from integers to vectors
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)


baseModel = ResNet152V2(weights="imagenet", include_top=False, input_tensor=input)

headModel = baseModel.output
headModel = keras.layers.GlobalAveragePooling2D()(headModel)
headModel = keras.layers.Flatten()(headModel)
headModel = keras.layers.Dense(512, activation="relu", 
                               kernel_regularizer=keras.regularizers.L2(l2_constant))(headModel)
headModel = keras.layers.Dense(num_classes, activation="softmax")(headModel)

model = keras.models.Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
   layer.trainable = False

opt = keras.optimizers.RMSprop(learning_rate=0.01)

model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

#Add Learning rate scheduler to model callbacks
callbacks=[keras.callbacks.LearningRateScheduler(step_decay)]

print("[INFO] training head...")

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs_head, 
          validation_split=0.1, validation_data=(x_test, y_test), 
          callbacks=callbacks)

for layer in baseModel.layers[0:]:
   layer.trainable = True

print("Re-compiling model...")

# Have to rebuild optimizer for model recompile
opt = keras.optimizers.SGD(learning_rate=0.005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
print("Model Compiled")

print("Fine-Tuning Final Model...")
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          validation_split=0.1, validation_data=(x_test, y_test), 
          callbacks=callbacks)

score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])