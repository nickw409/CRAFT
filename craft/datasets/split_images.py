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
  if not dir.exists():
    dir.mkdir()

  if read_csv(image_list, image_dict):
    for (image_name, label) in image_dict.items():
      match label:
        case 'Kanaa':
          categorized_image_dir = dir / 'Kanaa'
          if not categorized_image_dir.exists():
            categorized_image_dir.mkdir()
          shutil.copy(images_dir / image_name, categorized_image_dir / image_name)
        case 'Black_Mesa':
          categorized_image_dir = dir / 'Black_Mesa'
          if not categorized_image_dir.exists():
            categorized_image_dir.mkdir()
          shutil.copy(images_dir / image_name, categorized_image_dir / image_name)
        case 'Sosi':
          categorized_image_dir = dir / 'Sosi'
          if not categorized_image_dir.exists():
            categorized_image_dir.mkdir()
          shutil.copy(images_dir / image_name, categorized_image_dir / image_name)
        case 'Dogoszhi':
          categorized_image_dir = dir / 'Dogoszhi'
          if not categorized_image_dir.exists():
            categorized_image_dir.mkdir()
          shutil.copy(images_dir / image_name, categorized_image_dir / image_name)
        case 'Flagstaff':
          categorized_image_dir = dir / 'Flagstaff'
          if not categorized_image_dir.exists():
            categorized_image_dir.mkdir()
          shutil.copy(images_dir / image_name, categorized_image_dir / image_name)
        case 'Tusayan':
          categorized_image_dir = dir / 'Tusayan'
          if not categorized_image_dir.exists():
            categorized_image_dir.mkdir()
          shutil.copy(images_dir / image_name, categorized_image_dir / image_name)
        case 'Kayenta':
          categorized_image_dir = dir / 'Kayenta'
          if not categorized_image_dir.exists():
            categorized_image_dir.mkdir()
          shutil.copy(images_dir / image_name, categorized_image_dir / image_name)


def consolidate_image_data(image_dir):
  # go through dictionary key by key and search for image that matches
  # copy that image into either training, validation, or testing based on desired split
  # add key pair to new csv file in each dir depending on where it is placed
  # maintain proper ratios until all files have been copied
  
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
        print(row)
        csv_writer.writerow(row)
    csv_file.close()
  except IOError as e:
    sys.stderr.write(f"Error opening {filename} for writing\n {e}")
    return False
  return True


def read_csv(file_path, data_dict):
  try:
    with open(file_path, 'r') as csv_file:
      for row in csv.reader(csv_file):
        data_dict[row[0]] = row[1]
    csv_file.close()
    return True
  except IOError as e:
    sys.stderr.write(f"Error opening {file_path} for reading\n {e}")
    return False


image_dir = Path('.').resolve().parents[1] / 'image_data'
consolidate_image_data(image_dir)
image_list = Path('.').resolve().parents[1] / 'image_data' / 'image_list.csv'
categorize_images(image_list, 'tusayan_whiteware')