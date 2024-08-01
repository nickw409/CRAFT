import csv
import os
from pathlib import Path
import re
import sys


def consolidate_image_data(image_dir):
  # find image data csv files (done)
  # read all data into dictionary so no duplicates (done)
  # keeping adding to dictionary until all files are read (done)
  # for now write all pairs to new csv file (done)
  # go through dictionary key by key and search for image that matches
  # copy that image into either training, validation, or testing based in desired split
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
  except IOError:
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


curr_path = Path('.').resolve()
success = consolidate_image_data(curr_path.parents[1] / 'image_data')
print(success)