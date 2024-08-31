import cv2
import csv
from enum import Enum
import keras
import numpy as np
import os
from pathlib import Path
import random
import sys
import time

import split_images
try:
  sys.path.append(str(Path('.').resolve().parent))
  from misc.progress_bar import printProgressBar
except Exception as e:
  print(e)


class SherdType(Enum):
  Kanaa = 0
  Black_Mesa = 1
  Sosi = 2
  Dogoszhi = 3
  Flagstaff = 4
  Tusayan = 5
  Kayenta = 6


def create_training_test_sets(image_dir, training_split):
  """
  Since the training data we have for each class is disproportionate, we need to
  ensure that the testing set contains the same ratio of classes as found in the
  full image set.

  Args:
    image_dir: Full path to the image directory.
    training_split: floating point of percentage of data to use for training
  """
  image_count = [0] * 7
  training_image_count = [0] * 7
  data_dict = split_images.consolidate_image_data(image_dir)
  training_set_filename = image_dir / 'training_list.csv'
  test_set_filename = image_dir / 'test_list.csv'

  # Get the number of sherds per type 
  for val in data_dict.values():
    image_count[SherdType[val].value] += 1
  
  # Shuffle dictionary keys to create shuffled training and testing sets
  print(f'Shuffling dictionary keys')
  keys = list(data_dict.keys())
  random.shuffle(keys)
  # Open csv files for writing
  try:
    with open(training_set_filename, 'w') as training_csv, open(test_set_filename, 'w') as test_csv:
      training_writer = csv.writer(training_csv)
      test_writer = csv.writer(test_csv)
      rows_written = 0
      # Loop through dictionary using shuffled keys
      for key in keys:
        label = data_dict[key]
        idx = SherdType[label].value
        rows_written += 1
        # If current ratio less than training split add image to training set
        if training_image_count[idx] / image_count[idx] < training_split:
          training_writer.writerow([key, label])
          training_image_count[idx] += 1
        else:
          test_writer.writerow([key, label])
        if rows_written % 70 == 0:
          printProgressBar(rows_written, len(keys), 'Progress', 'Complete', length=50)
          time.sleep(0.1)
      printProgressBar(len(keys), len(keys), 'Progress', 'Complete', length=50)
          
  except IOError as e:
    sys.stderr.write(f"Error opening file for writing\n {e}")
    return False
  
  split_images.categorize_images(image_dir, 'tusayan_whiteware')
  return True


def load_data(image_dimension, training_split, verbose=-1):
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

image_dir = Path('.').resolve().parents[1] / 'image_data'
create_training_test_sets(image_dir, 0.8)