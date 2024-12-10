from PySide6.QtWidgets import (QLabel, QWidget, QGridLayout, QLineEdit, QPushButton, QFileDialog, QCheckBox )
from PySide6.QtCore import Qt, Signal

chosen_settings = {"saved_img_dir":"",
                   "saved_img_csv_file":"",
                   "img_latitude": "",
                   "img_longitude": "",
                   "classify_bool": False,
                   "upload_bool": False,
                   "training_data_dir":""}

class Settings(QWidget):
    # create signals to pass data to other files
    save_settings_signal = Signal()

    def __init__(self):
        super().__init__()
        
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(4, 1)

        # option 1: directory for saved images
        self.saved_image_dir_message = QLabel("Select an existing directory to save images in: ")
        self.saved_image_dir_input = QLineEdit()
        self.saved_image_dir_input.setReadOnly(True)
        saved_image_dir_button = QPushButton("Search Files")
        saved_image_dir_button.clicked.connect(self.directory_dialog)

            # option 1: layout
        grid.addWidget(self.saved_image_dir_message, 1, 0, 1, 5)
        grid.addWidget(self.saved_image_dir_input, 2, 0, 1, 4)
        grid.addWidget(saved_image_dir_button, 2, 4)

        # option 2: directory to save csv file data
        # TODO: fix. currently only allows existing csv files to be written to
        csv_file_message = QLabel("Select an existing file to save CSV data to: ")
        self.csv_file_input = QLineEdit()
        self.csv_file_input.setReadOnly(True)
        csv_file_button = QPushButton("Search Files")
        csv_file_button.clicked.connect(self.csv_dialog)

            # option 2: layout
        grid.addWidget(csv_file_message, 3, 0, 1, 5)
        grid.addWidget(self.csv_file_input, 4, 0, 1, 4)
        grid.addWidget(csv_file_button, 4, 4)

        # option 3: change latitude and longitude
        latitude_message = QLabel("Enter latitude: ")
        self.latitude_input = QLineEdit()
        longitude_message = QLabel("Enter longitude: ")
        self.longitude_input = QLineEdit()

            # option 3: layout
        grid.addWidget(latitude_message, 5, 0, 1, 5)
        grid.addWidget(self.latitude_input, 6, 0, 1, 3)
        grid.addWidget(longitude_message, 7, 0, 1, 5)
        grid.addWidget(self.longitude_input, 8, 0, 1, 3)

        # option 4: classify images?
        classify_checkbox_message = QLabel("Classify images with deep learning model: ")
        self.classify_checkbox = QCheckBox()

            # option 4: layout
        grid.addWidget(classify_checkbox_message, 9, 0, 1, 2)
        grid.addWidget(self.classify_checkbox, 9, 2)

        # option 5: upload images to cloud storage?
        upload_checkbox_message = QLabel("Upload saved images to cloud storage: ")
        self.upload_checkbox = QCheckBox()

            # option 5: layout
        grid.addWidget(upload_checkbox_message, 10, 0, 1, 2)
        grid.addWidget(self.upload_checkbox, 10, 2)

        # option 6: add to training data?
        training_data_message = QLabel("Select an existing csv file of training data to add images to: ")
        self.training_data_input = QLineEdit()
        self.training_data_input.setReadOnly(True)
        training_data_button = QPushButton("Search Files")
        training_data_button.clicked.connect(self.training_csv_dialog)

            # option 6: layout
        grid.addWidget(training_data_message, 11, 0, 1, 5)
        grid.addWidget(self.training_data_input, 12, 0, 1, 4)
        grid.addWidget(training_data_button, 12, 4)

        # submit button
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_settings)
        grid.addWidget(save_button, 13, 2, 1, 1)

        # set the layout
        grid.setRowStretch(0, 1)
        grid.setRowStretch(14, 1)
        self.setLayout(grid)

    def save_settings(self):
        self.save_settings_signal.emit()

        # save settings data
        chosen_settings["saved_img_dir"] = self.saved_image_dir_input.text()
        chosen_settings["saved_img_csv_file"] = self.csv_file_input.text()
        chosen_settings["img_latitude"] = self.latitude_input.text()
        chosen_settings["img_longitude"] = self.longitude_input.text()
        chosen_settings["classify_bool"] = self.classify_checkbox.isChecked()
        chosen_settings["upload_bool"] = self.upload_checkbox.isChecked()
        chosen_settings["training_data_dir"] = self.training_data_input.text()

    def directory_dialog(self):
        image_dir = QFileDialog.getExistingDirectory(self, str("Open Directory"), "/home", QFileDialog.ShowDirsOnly)
        self.saved_image_dir_input.setText(image_dir)
    
    def csv_dialog(self):
        file = QFileDialog.getOpenFileName(self, str("Open CSV"), "/home", str("CSV Files (*.csv)"))
        self.csv_file_input.setText(file[0])

    def training_csv_dialog(self):
        file = QFileDialog.getOpenFileName(self, str("Open CSV"), "/home", str("CSV Files (*.csv)"))
        self.training_data_input.setText(file[0])       