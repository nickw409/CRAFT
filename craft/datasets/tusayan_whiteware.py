import cv2
import csv
import keras
import numpy as np
import os
from pathlib import Path


def load_data(training_split, image_dimension, verbose=-1):
  """
  Args:
    training_split: a tuple (training, validation, testing) which takes in
    doubles in the range 0.0-1.0. Must add up to 1.0
  """
  # initialize the list of features and labels
  data = []
  labels = []
  i=0

  imagelist_path = Path('.').resolve()
  imagelist_path = imagelist_path.parents[1] / 'image_data' / 'image_list.csv'
  images_dir = imagelist_path.parent / 'images'
  print(imagelist_path)
  print(images_dir)

  try:
    with open(imagelist_path, 'r') as csv_file:
      for row in csv.reader(csv_file):
        # Load image 
        image = cv2.imread(images_dir / row[0])
        # resize image, convert image to grayscale, then back to original color scheme
        image = cv2.cvtColor(
            cv2.cvtColor(
              cv2.resize(image, (image_dimension, image_dimension),
                interpolation=cv2.INTER_AREA),
              cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR)
        # Convert image to keras array format
        image=keras.utils.img_to_array(image)
        data.append(image)
        labels.append(row[1])
  except IOError as e:
    print(f"Error reading file {imagelist_path}: {e}")

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


