import csv
import os
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
  
  for (dir, subdirs, files) in os.walk(image_dir):
    match = re.search(set_pattern, dir)
    if match:
      for filename in files:
        match = re.search(filetype_pattern, filename)
        if match:
          read_csv(filename, data_dict)
  
  full_imagelist_path = os.path.join(image_dir, "image_list.csv")
  try:
    with open(full_imagelist_path, 'w') as csv_file:
      csv_writer = csv.writer(csv_file)
      for row in data_dict:
        csv_writer.writerow(row)
  except IOError:
    sys.stderr.write(f"Error opening {filename} for writing\n")
    return False
  return True


def read_csv(filename, data_dict):
  try:
    with open(filename, 'r') as csv_file:
      for row in csv.reader(csv_file):
        data_dict[row[0]] = row[1]
    csv_file.close()
    return True
  except IOError:
    sys.stderr.write(f"Error opening {filename} for reading\n")
    return False