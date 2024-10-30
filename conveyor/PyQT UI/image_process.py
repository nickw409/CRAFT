from PySide6.QtWidgets import (
    QWidget, QGridLayout, QPushButton, QHBoxLayout)
from PySide6.QtCore import Qt
from cameraView import CameraView
from camera_thread import CameraThread
from capturedImageView import CapturedImageView

class ImageProcess(QWidget):

    def __init__(self):
        super().__init__()
        grid = QGridLayout()

        # set up camera view and connect signal
        self.camera_label = CameraView()
        grid.addWidget(self.camera_label, 0, 0)
        self.cameraThread = CameraThread()
        self.cameraThread.video_view_signal.connect(self.camera_label.setImage)

        # display captured image
        self.captured_image = CapturedImageView()
        grid.addWidget(self.captured_image, 0, 1)
        self.cameraThread.image_captured_signal.connect(self.captured_image.setImage)
        
        # create start and stop buttons
        buttonWidget = QWidget()
        button_box = QHBoxLayout()
        buttonWidget.setLayout(button_box)
        self.startButton = QPushButton("Start")
        self.stopButton = QPushButton("Stop")
        button_box.addWidget(self.startButton)
        button_box.addWidget(self.stopButton)
        grid.addWidget(buttonWidget, 1, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        # connect video functions to buttons
        self.startButton.clicked.connect(self.cameraThread.start_thread)
        self.stopButton.clicked.connect(self.stop_thread)
        
        # setting the layout
        self.setLayout(grid)

    def stop_thread(self):
        self.cameraThread.stop = True