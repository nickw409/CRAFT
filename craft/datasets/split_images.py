import csv
import os
from pathlib import Path
import re
import shutil
import sys


def categorize_images(image_dir, dir_name):
  """
  Args:
    image_dir: Full path to the image directory.
    dir_name: String of the desired name to give to parent directory where 
    images are stored.
  """
  images_dir = image_dir / 'images'
  categorized_dir = image_dir / dir_name
  training_dir = categorized_dir / 'Training'
  test_dir = categorized_dir / 'Testing'
  if not categorized_dir.exists():
    categorized_dir.mkdir()
  if not training_dir.exists():
    training_dir.mkdir()
  if not test_dir.exists():
    test_dir.mkdir()
  
  # Find csv files for training and testing dynamically
  files = os.listdir(image_dir)

  # Copy all training images into training dir
  for file in files:
    training_match = re.search('training', file)
    test_match = re.search('test', file)
    if training_match:
      training_list = image_dir / training_match.string
      copy_images(training_list, images_dir, training_dir)
    # Copy all test images into test dir
    if test_match:
      test_list = image_dir / test_match.string
      copy_images(test_list, images_dir, test_dir)



def consolidate_image_data(image_dir): 
  """
  Image data was originally split into set directories for cross-fold 
  validation. Labels were stored in csv files for each set along with the file
  name of the image. This function goes through every set directory and copies
  the image file names and labels to a main csv file that contains all data.

  Args:
    image_dir: Full path to the image directory.  
  Returns:
    A dictionary containing every image file name and corresponding label
  """
  # Search all subdirectories for directories named Set_
  set_pattern = "Set_"
  filetype_pattern = ".csv"
  data_dict = {}

  for (dir, subdirs, files) in os.walk(image_dir):
    match = re.search(set_pattern, dir)
    if match:
      for filename in files:
        match = re.search(filetype_pattern, filename)
        if match:
          file_path = Path(dir) / filename
          read_csv(file_path, data_dict)
  
  full_imagelist_path = os.path.join(image_dir, "image_list.csv")
  try:
    with open(full_imagelist_path, 'w') as csv_file:
      csv_writer = csv.writer(csv_file)
      for row in data_dict.items():
        csv_writer.writerow(row)
  except IOError as e:
    sys.stderr.write(f"Error opening {filename} for writing\n {e}")
    return False
  return data_dict


def copy_images(image_list, source, destination):
  data_dict = {}
  read_csv(image_list, data_dict)
  for (image_name, label) in data_dict.items():      
    categorized_image_dir = destination / label
    if not categorized_image_dir.exists():
      categorized_image_dir.mkdir()
    shutil.copy(source / image_name, categorized_image_dir / image_name)
  

def read_csv(file_path, data_dict):
  """
  Read in csv file data into dictionary.
  Args: 
    file_path: Full path to csv file.
    data_dict: Python dictionary to hold data.
  """
  try:
    with open(file_path, 'r') as csv_file:
      for row in csv.reader(csv_file):
        data_dict[row[0]] = row[1]
    return True
  except IOError as e:
    sys.stderr.write(f"Error opening {file_path} for reading\n {e}")
    return False
