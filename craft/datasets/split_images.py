import csv
import os
from pathlib import Path
import re
import shutil
import sys


def categorize_images(image_list, dir_name):
  """
  Args:
    image_list: Path to the csv file containing image list information.
    dir_name: String of the desired name to give to parent directory where 
    images are stored.
  """
  image_dict = {}
  images_dir = Path('.').resolve().parents[1] / 'image_data' / 'images'
  dir = images_dir.parents[0] / dir_name
  training_dir = dir / 'Training'
  test_dir = dir / 'Testing'
  if not dir.exists():
    dir.mkdir()
  if not training_dir.exists():
    training_dir.mkdir()
  if not test_dir.exists():
    test_dir.mkdir()

  if read_csv(image_list, image_dict):
    for (image_name, label) in image_dict.items():
      match label:
        case 'Kanaa':
          sherd_type = 'Kanaa'
        case 'Black_Mesa':
          sherd_type = 'Black_Mesa'
        case 'Sosi':
          sherd_type = 'Sosi'
        case 'Dogoszhi':
          sherd_type = 'Dogoszhi'
        case 'Flagstaff':
          sherd_type = 'Flagstaff'
        case 'Tusayan':
          sherd_type = 'Tusayan'
        case 'Kayenta':
          sherd_type = 'Kayenta'
      
      categorized_image_dir = dir / sherd_type
      if not categorized_image_dir.exists():
        categorized_image_dir.mkdir()
      shutil.copy(images_dir / image_name, categorized_image_dir / image_name)


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
  print(image_dir)

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


image_dir = Path('.').resolve().parents[1] / 'image_data'
consolidate_image_data(image_dir)
#image_list = Path('.').resolve().parents[1] / 'image_data' / 'image_list.csv'
#categorize_images(image_list, 'tusayan_whiteware')