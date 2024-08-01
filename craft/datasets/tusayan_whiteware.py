import cv2
from csv import reader
import keras
import numpy as np
import os


def load_data(set_number, image_dimension, verbose=-1):
  # Location of set csv files
  set_directory = os.getcwd() + "../../image_data"
  #Define directories to use based on set number for loading data, saving data
  train_dataset = set_directory + "/Set_" + set_number +"/train_" + set_number + ".csv"
  test_dataset = set_directory + "/Set_" + set_number +"/test_" + set_number + ".csv"
  # initialize the list of features and labels
  data = []
  labels = []
  i=0

  # Get lists of train images, types from file
  print("[INFO] loading images...")
  with open(train_dataset, 'r') as read_obj:
      # pass the file object to reader() to get the reader object
      csv_reader = reader(read_obj)
      # Pass reader object to list() to get a list of lists
      train_data_list = list(csv_reader)

  # close file
  read_obj.close()

    # same for test images, types
  with open(test_dataset, 'r') as read_obj:
      # pass the file object to reader() to get the reader object
      csv_reader = reader(read_obj)
      # Pass reader object to list() to get a list of lists
      test_data_list = list(csv_reader)

  read_obj.close()

  # loop over the input images
  for image_input in imageData:
      # load the image
      images_dir = os.getcwd() + "../../image_data/images/"
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


def get_images()